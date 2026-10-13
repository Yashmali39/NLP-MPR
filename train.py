"""
train.py

Full, reproducible fine-tuning pipeline for DistilBERT on the PAWS paraphrase dataset.
Supports local execution (Apple Silicon MPS, CUDA, CPU) with:
- Stratified subsetting preserving class distribution
- Class-weighted Cross-Entropy loss to prevent majority-class collapse
- Epoch-by-epoch validation monitoring (Accuracy, Macro-F1, per-class recall)
- Best-checkpoint selection based on Validation Macro-F1
- Rigorous held-out test evaluation (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, ROC-AUC)
- Mandatory sanity-check test suite (Identical text, Lexical paraphrase, Reversed scramble, Distinct topics)
- Serialization of authentic metrics and candidate artifacts
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

from src.preprocessing import clean_text, validate_paws_records, check_split_leakage, ID2LABEL, LABEL2ID
from src.predict import get_optimal_device
from src.evaluation import (
    compute_classification_metrics,
    evaluate_tfidf_baseline,
    evaluate_bertscore,
    save_evaluation_artifacts
)


class TextPairDataset(Dataset):
    """PyTorch Dataset wrapper for paired sequence classification."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Perform reproducible stratified sampling based on the binary label column.
    Guarantees balanced representation without distorting the underlying class proportions.
    """
    if n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    proportions = df["label"].value_counts(normalize=True)
    sampled_splits = []
    for label_val, prop in proportions.items():
        sub_df = df[df["label"] == label_val]
        sample_count = int(np.rint(n * prop))
        sample_count = min(sample_count, len(sub_df))
        sampled_splits.append(sub_df.sample(n=sample_count, random_state=seed))

    combined = pd.concat(sampled_splits).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(combined) > n:
        combined = combined.iloc[:n].reset_index(drop=True)
    return combined


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on PAWS dataset.")
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased",
                        help="Base pretrained model.")
    parser.add_argument("--subset_train", type=int, default=8000,
                        help="Number of training samples (use -1 for full split).")
    parser.add_argument("--subset_val", type=int, default=1000,
                        help="Number of validation samples (use -1 for full split).")
    parser.add_argument("--subset_test", type=int, default=1000,
                        help="Number of held-out test samples (use -1 for full split).")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per training step.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum token sequence length.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned_distilbert_candidate",
                        help="Directory to save the candidate fine-tuned model checkpoint.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save evaluation artifacts.")
    return parser.parse_args()


def load_and_preprocess_paws(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load official PAWS labeled_final splits, validate records, clean data, and subset with stratification.
    """
    print(f"[*] Loading PAWS (labeled_final) from Hugging Face Datasets...")
    try:
        dataset = load_dataset("google-research-datasets/paws", "labeled_final")
    except Exception:
        dataset = load_dataset("paws", "labeled_final")

    train_raw = dataset["train"].to_pandas()
    val_raw = dataset["validation"].to_pandas()
    test_raw = dataset["test"].to_pandas()

    print(f"[*] Raw records -> Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    train_df, train_stats = validate_paws_records(train_raw)
    val_df, val_stats = validate_paws_records(val_raw)
    test_df, test_stats = validate_paws_records(test_raw)

    print(f"[*] Cleaned records -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Check for split leakage
    leakage = check_split_leakage(train_df, test_df)
    print(f"[*] Train-Test Leakage Check: {leakage['exact_overlap_count']} exact overlaps detected.")

    # Apply stratified sampling to preserve class distribution
    if args.subset_train > 0 and args.subset_train < len(train_df):
        train_df = stratified_subset(train_df, args.subset_train, seed=args.seed)
    if args.subset_val > 0 and args.subset_val < len(val_df):
        val_df = stratified_subset(val_df, args.subset_val, seed=args.seed)
    if args.subset_test > 0 and args.subset_test < len(test_df):
        test_df = stratified_subset(test_df, args.subset_test, seed=args.seed)

    print(f"[*] Active training subset -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"    Train class balance: {train_df['label'].value_counts().to_dict()}")
    print(f"    Val class balance:   {val_df['label'].value_counts().to_dict()}")
    print(f"    Test class balance:  {test_df['label'].value_counts().to_dict()}")

    return train_df, val_df, test_df


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int
) -> TextPairDataset:
    """Encode dataframe sentence pairs into a PyTorch TextPairDataset."""
    s1_list = df["sentence1"].tolist()
    s2_list = df["sentence2"].tolist()
    labels = df["label"].tolist()

    encodings = tokenizer(
        s1_list,
        s2_list,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return TextPairDataset(encodings, labels)


def train_model(
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    loss_fn: torch.nn.Module,
    output_dir: str
) -> Tuple[Dict[str, List], int]:
    """
    Execute PyTorch training loop across epochs, tracking validation macro-F1
    and saving the best checkpoint based on validation performance.
    """
    history = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_macro_f1s": [],
        "val_recall_0s": [],
        "val_recall_1s": []
    }
    best_macro_f1 = -1.0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_trues.extend(labels.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = float(accuracy_score(val_trues, val_preds))
        val_macro_f1 = float(f1_score(val_trues, val_preds, average="macro", zero_division=0))
        val_rec_0 = float(recall_score(val_trues, val_preds, pos_label=0, zero_division=0))
        val_rec_1 = float(recall_score(val_trues, val_preds, pos_label=1, zero_division=0))

        print(f"[*] Epoch {epoch}/{epochs} Summary: Train Loss = {avg_train_loss:.4f} | "
              f"Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.4f} | "
              f"Val Macro-F1 = {val_macro_f1:.4f} | Rec(0) = {val_rec_0:.4f} | Rec(1) = {val_rec_1:.4f}")

        history["epochs"].append(epoch)
        history["train_losses"].append(round(avg_train_loss, 4))
        history["val_losses"].append(round(avg_val_loss, 4))
        history["val_accuracies"].append(round(val_acc, 4))
        history["val_macro_f1s"].append(round(val_macro_f1, 4))
        history["val_recall_0s"].append(round(val_rec_0, 4))
        history["val_recall_1s"].append(round(val_rec_1, 4))

        # Checkpoint selection based on Validation Macro-F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            print(f"    [+] New Best Validation Macro-F1: {best_macro_f1:.4f}. Saving best checkpoint to '{output_dir}'...")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"\n[*] Training Complete. Best Validation Macro-F1: {best_macro_f1:.4f} achieved at Epoch {best_epoch}.")
    return history, best_epoch


def evaluate_on_test_data(
    model: DistilBertForSequenceClassification,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int], List[float], float]:
    """
    Run full inference on held-out test dataset, measuring predictions, probabilities, and latency.
    """
    model.eval()
    all_preds: List[int] = []
    all_trues: List[int] = []
    all_probs: List[float] = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Held-Out Test Inference", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_trues.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / len(all_preds)) * 1000.0

    return all_trues, all_preds, all_probs, avg_latency_ms


def run_sanity_checks(
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    test_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Execute mandatory sanity tests (A, B, C, D) and held-out dataset pairs.
    """
    model.eval()

    sanity_cases = [
        {
            "id": "Test A",
            "name": "Identical Text",
            "text_a": "Artificial intelligence can process natural language.",
            "text_b": "Artificial intelligence can process natural language.",
            "expected_label": "Paraphrase",
            "expected_id": 1
        },
        {
            "id": "Test B",
            "name": "Clear Lexical Paraphrase",
            "text_a": "The company announced a new product yesterday.",
            "text_b": "Yesterday, the company revealed its latest product.",
            "expected_label": "Paraphrase",
            "expected_id": 1
        },
        {
            "id": "Test C",
            "name": "Reversed Subject/Object",
            "text_a": "The police officer arrested the criminal.",
            "text_b": "The criminal arrested the police officer.",
            "expected_label": "Not Paraphrase",
            "expected_id": 0
        },
        {
            "id": "Test D",
            "name": "Different Meanings",
            "text_a": "The weather is sunny today.",
            "text_b": "The database server requires additional memory.",
            "expected_label": "Not Paraphrase",
            "expected_id": 0
        }
    ]

    # Add 4 untouched samples from the test dataframe if available
    if test_df is not None and len(test_df) >= 4:
        # Find 2 true positive and 2 true negative samples
        pos_samples = test_df[test_df["label"] == 1].head(2)
        neg_samples = test_df[test_df["label"] == 0].head(2)
        for idx, row in pos_samples.iterrows():
            sanity_cases.append({
                "id": f"PAWS-Pos-{row['id']}",
                "name": "PAWS Ground-Truth Paraphrase",
                "text_a": row["sentence1"],
                "text_b": row["sentence2"],
                "expected_label": "Paraphrase",
                "expected_id": 1
            })
        for idx, row in neg_samples.iterrows():
            sanity_cases.append({
                "id": f"PAWS-Neg-{row['id']}",
                "name": "PAWS Ground-Truth Non-Paraphrase",
                "text_a": row["sentence1"],
                "text_b": row["sentence2"],
                "expected_label": "Not Paraphrase",
                "expected_id": 0
            })

    results = []
    print("\n" + "=" * 80)
    print("MANDATORY SANITY TEST SUITE EVALUATION")
    print("=" * 80)

    for case in sanity_cases:
        inputs = tokenizer(
            case["text_a"],
            case["text_b"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        is_correct = (pred_id == case["expected_id"])

        record = {
            "test_id": case["id"],
            "name": case["name"],
            "text_a": case["text_a"][:70] + ("..." if len(case["text_a"]) > 70 else ""),
            "text_b": case["text_b"][:70] + ("..." if len(case["text_b"]) > 70 else ""),
            "expected_label": case["expected_label"],
            "predicted_label": pred_label,
            "prob_not_paraphrase": round(float(probs[0]), 4),
            "prob_paraphrase": round(float(probs[1]), 4),
            "is_correct": is_correct
        }
        results.append(record)

        mark = "PASS [OK]" if is_correct else "FAIL [X]"
        print(f"[{case['id']}: {case['name']}] -> {mark}")
        print(f"  Expected : {case['expected_label']} | Predicted: {pred_label}")
        print(f"  P(Not Paraphrase) = {probs[0]:.4f} | P(Paraphrase) = {probs[1]:.4f}")

    passed_count = sum(1 for r in results if r["is_correct"])
    print(f"\nSanity Tests Passed: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
    print("=" * 80)
    return results


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    print("=" * 80)
    print("DISTILBERT FINE-TUNING PIPELINE: PLAGAWARE AI PARAPHRASE DETECTION")
    print("=" * 80)
    print(f"Base Pretrained Model : {args.model_name}")
    print(f"Target Subsets        : Train={args.subset_train}, Val={args.subset_val}, Test={args.subset_test}")
    print(f"Training Hyperparams  : Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}")
    print(f"Candidate Output Dir  : {args.output_dir}")

    device = get_optimal_device()
    print(f"Hardware Device       : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Apple Silicon / Host'})")

    # Step 1: Load and clean PAWS dataset with stratification
    train_df, val_df, test_df = load_and_preprocess_paws(args)

    # Step 2: Compute Balanced Class Weights for Cross-Entropy
    train_counts = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    w0 = total_samples / (2.0 * train_counts[0])
    w1 = total_samples / (2.0 * train_counts[1])
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"[*] Class Weights Applied: Class 0 = {w0:.4f}, Class 1 = {w1:.4f}")

    # Step 3: Initialize Tokenizer and Model
    print(f"\n[*] Initializing Tokenizer and Model from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    # Step 4: Tokenize Splits
    print("[*] Tokenizing text pairs with paired [CLS] s1 [SEP] s2 [SEP] formatting...")
    train_dataset = tokenize_dataframe(train_df, tokenizer, args.max_length)
    val_dataset = tokenize_dataframe(val_df, tokenizer, args.max_length)
    test_dataset = tokenize_dataframe(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Step 5: Optimizer and Schedule
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Step 6: Fine-Tuning Loop with Best-Checkpoint Selection
    print(f"\n[*] Beginning Fine-Tuning ({args.epochs} epochs, {total_steps} total optimization steps)...")
    training_start = time.perf_counter()
    history, best_epoch = train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        loss_fn=loss_fn,
        output_dir=args.output_dir
    )
    training_duration_sec = round(time.perf_counter() - training_start, 2)
    print(f"[*] Training finished in {training_duration_sec} seconds ({training_duration_sec / 60.0:.2f} mins).")

    # Step 7: Load Best Checkpoint for Evaluation
    print(f"\n[*] Loading best saved checkpoint from '{args.output_dir}' (Epoch {best_epoch})...")
    best_model = DistilBertForSequenceClassification.from_pretrained(args.output_dir)
    best_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    best_model.to(device)

    # Step 8: Evaluate Best Candidate on Untouched Held-Out Test Set
    print("\n[*] Evaluating best candidate model on untouched held-out test split...")
    y_true, y_pred, y_probs, avg_latency_ms = evaluate_on_test_data(best_model, test_loader, device)

    metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    print(f"\n[HELD-OUT TEST RESULTS]")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision (Class 1): {metrics['binary_metrics']['precision']:.4f}")
    print(f"Recall (Class 1)   : {metrics['binary_metrics']['recall']:.4f}")
    print(f"F1-Score (Class 1) : {metrics['binary_metrics']['f1_score']:.4f}")
    print(f"Macro F1-Score     : {metrics['macro_metrics']['f1_score']:.4f}")
    print(f"Weighted F1-Score  : {metrics['weighted_f1']:.4f}")
    print(f"ROC-AUC            : {metrics['roc_auc']}")
    print(f"Inference Latency  : {avg_latency_ms:.2f} ms / sample")

    print("\n[CLASSIFICATION REPORT]")
    print(metrics["classification_report_text"])

    # Step 9: TF-IDF Baseline Comparison on Identical Split
    print("[*] Evaluating TF-IDF + Logistic Regression baseline on identical test split...")
    train_pairs = list(zip(train_df["sentence1"].tolist(), train_df["sentence2"].tolist()))
    test_pairs = list(zip(test_df["sentence1"].tolist(), test_df["sentence2"].tolist()))
    tfidf_baseline_results = evaluate_tfidf_baseline(
        train_pairs, train_df["label"].tolist(),
        test_pairs, test_df["label"].tolist()
    )
    print(f"TF-IDF Baseline Results: {tfidf_baseline_results}")

    # Step 10: Supplementary BERTScore on Sample
    print("[*] Evaluating supplementary BERTScore on test sample...")
    sample_size = min(100, len(test_df))
    bertscore_res = evaluate_bertscore(
        test_df["sentence1"].head(sample_size).tolist(),
        test_df["sentence2"].head(sample_size).tolist(),
        model_type="distilbert-base-uncased"
    )
    print(f"BERTScore Results: {bertscore_res}")

    # Step 11: Execute Mandatory Sanity Test Suite
    sanity_results = run_sanity_checks(best_model, best_tokenizer, device, test_df)

    # Step 12: Compile Comprehensive Payload and Save Candidate Artifacts
    cm_arr = np.array([
        [metrics["confusion_matrix"]["true_negatives"], metrics["confusion_matrix"]["false_positives"]],
        [metrics["confusion_matrix"]["false_negatives"], metrics["confusion_matrix"]["true_positives"]]
    ])

    full_results_payload = {
        "dataset": "PAWS (labeled_final)",
        "base_model": args.model_name,
        "device": device.type,
        "training_metadata": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "epochs_trained": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "training_duration_seconds": training_duration_sec,
            "loss_history": history
        },
        "distilbert_evaluation": {
            "accuracy": metrics["accuracy"],
            "binary_precision": metrics["binary_metrics"]["precision"],
            "binary_recall": metrics["binary_metrics"]["recall"],
            "binary_f1": metrics["binary_metrics"]["f1_score"],
            "macro_f1": metrics["macro_metrics"]["f1_score"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc": metrics["roc_auc"],
            "per_class_metrics": metrics["per_class_metrics"],
            "class_distributions": metrics["class_distributions"],
            "inference_latency_ms": round(avg_latency_ms, 2),
            "confusion_matrix": metrics["confusion_matrix"]
        },
        "tfidf_baseline_comparison": tfidf_baseline_results,
        "supplementary_bertscore": bertscore_res,
        "sanity_test_results": sanity_results
    }

    # Save artifacts into candidate results folder or results/
    saved_artifacts = save_evaluation_artifacts(
        metrics_data=full_results_payload,
        report_text=metrics["classification_report_text"],
        confusion_mat=cm_arr,
        training_history=history,
        output_dir=args.results_dir
    )

    print("\n[*] Evaluation Artifacts Generated:")
    for k, v in saved_artifacts.items():
        print(f"    - {k}: {v}")

    print("\n" + "=" * 80)
    print("DISTILBERT CANDIDATE FINE-TUNING AND EVALUATION COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
