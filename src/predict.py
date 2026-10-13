"""
src/predict.py

Inference pipeline for the fine-tuned DistilBERT paraphrase classification model.
Enforces strict model checkpoint verification and provides comprehensive comparative
analysis combining deep neural predictions with surface lexical metrics.
"""

import os
import time
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.preprocessing import clean_text, validate_input_pair, ID2LABEL, LABEL2ID
from src.similarity import (
    compute_tfidf_cosine_similarity,
    compute_jaccard_similarity,
    extract_overlapping_phrases,
    explain_lexical_vs_semantic,
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the fine-tuned DistilBERT checkpoint directory or required weight files are missing."""
    pass


def get_optimal_device() -> torch.device:
    """
    Select the optimal execution device: CUDA GPU -> Apple Silicon MPS -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_checkpoint_available(model_dir: str = "models/fine_tuned_distilbert") -> bool:
    """
    Check if a valid fine-tuned checkpoint exists at the given path.
    Requires config.json and at least one weight file (model.safetensors or pytorch_model.bin).
    """
    if not os.path.isdir(model_dir):
        return False

    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    has_weights = (
        os.path.isfile(os.path.join(model_dir, "model.safetensors"))
        or os.path.isfile(os.path.join(model_dir, "pytorch_model.bin"))
    )
    return has_config and has_weights


def load_fine_tuned_model(
    model_dir: str = "models/fine_tuned_distilbert",
    device: Optional[torch.device] = None
) -> Tuple[DistilBertForSequenceClassification, AutoTokenizer, torch.device]:
    """
    Load the fine-tuned DistilBERT model and its matching tokenizer from disk.
    Raises ModelNotFoundError if the fine-tuned checkpoint is missing.
    """
    if not is_checkpoint_available(model_dir):
        raise ModelNotFoundError(
            f"Fine-tuned DistilBERT model checkpoint was not found at '{model_dir}'.\n"
            "In accordance with project integrity requirements, an untrained base model will NOT "
            "be silently substituted. Please fine-tune the model using the provided Google Colab notebook "
            "('notebooks/plagiarism_paraphrase_finetuning.ipynb') or local training script ('train.py'), "
            "and place the saved checkpoint into 'models/fine_tuned_distilbert/'."
        )

    if device is None:
        device = get_optimal_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_paraphrase(
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    text_a: str,
    text_b: str,
    device: Optional[torch.device] = None,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Perform sequence classification inference on a pair of text passages.
    Returns:
        dict containing predicted class, class probabilities, confidence, and inference latency.
    """
    is_valid, err_msg = validate_input_pair(text_a, text_b)
    if not is_valid:
        raise ValueError(err_msg)

    if device is None:
        device = next(model.parameters()).device

    start_time = time.perf_counter()

    clean_a = clean_text(text_a)
    clean_b = clean_text(text_b)

    # Verbatim identical text identity condition
    if clean_a.lower() == clean_b.lower():
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "predicted_label": ID2LABEL[1],  # "Paraphrase"
            "predicted_label_id": 1,
            "confidence": 1.0,
            "probabilities": {
                ID2LABEL[0]: 0.0,
                ID2LABEL[1]: 1.0
            },
            "latency_ms": round(latency_ms, 2)
        }

    # DistilBERT paired tokenization: [CLS] text_a [SEP] text_b [SEP]
    inputs = tokenizer(
        clean_a,
        clean_b,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    prob_not_paraphrase = float(probs[0])
    prob_paraphrase = float(probs[1])

    pred_id = int(probs.argmax())
    pred_label = ID2LABEL[pred_id]
    confidence = prob_paraphrase if pred_id == 1 else prob_not_paraphrase

    return {
        "predicted_label": pred_label,
        "predicted_label_id": pred_id,
        "confidence": confidence,
        "probabilities": {
            ID2LABEL[0]: prob_not_paraphrase,
            ID2LABEL[1]: prob_paraphrase
        },
        "latency_ms": round(latency_ms, 2)
    }


def analyze_text_pair(
    text_a: str,
    text_b: str,
    model: Optional[DistilBertForSequenceClassification] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Complete comparative analysis between Text A and Text B.
    Computes:
    1. Lexical surface similarity (TF-IDF Cosine Similarity)
    2. Jaccard word-level overlap
    3. Contiguous overlapping phrases
    4. Fine-tuned DistilBERT semantic prediction (if model and tokenizer provided)
    5. Comprehensive pedagogical explanation of lexical vs. semantic results
    """
    clean_a = clean_text(text_a)
    clean_b = clean_text(text_b)

    # 1. Surface lexical similarity
    tfidf_score = compute_tfidf_cosine_similarity(clean_a, clean_b)
    jaccard_score = compute_jaccard_similarity(clean_a, clean_b)
    overlapping_phrases = extract_overlapping_phrases(clean_a, clean_b)

    # 2. Semantic model inference
    semantic_result = None
    if model is not None and tokenizer is not None:
        semantic_result = predict_paraphrase(model, tokenizer, clean_a, clean_b, device=device)
        semantic_label = semantic_result["predicted_label"]
        semantic_prob = semantic_result["probabilities"]["Paraphrase"]
    else:
        semantic_label = "Model Not Loaded"
        semantic_prob = 0.0

    # 3. Comparative interpretation
    explanation = explain_lexical_vs_semantic(tfidf_score, semantic_label, semantic_prob)

    return {
        "text_a": clean_a,
        "text_b": clean_b,
        "tfidf_cosine_similarity": round(tfidf_score, 4),
        "jaccard_similarity": round(jaccard_score, 4),
        "overlapping_phrases": overlapping_phrases,
        "overlapping_phrases_count": len(overlapping_phrases),
        "semantic_prediction": semantic_result,
        "explanation": explanation
    }
