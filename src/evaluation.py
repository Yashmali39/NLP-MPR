"""
src/evaluation.py

Comprehensive evaluation framework:
- Accuracy, Precision, Recall, F1 (binary and macro)
- Confusion matrix plotting and serialization
- Training/validation loss curves plotting
- TF-IDF baseline evaluation on the identical test split
- Supplementary BERTScore evaluation
- Serialization of actual metrics and artifacts into results/
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import ID2LABEL


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute rigorous binary, macro, weighted, and per-class classification metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Binary metrics (Class 1 = Paraphrase)
    prec_bin = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    rec_bin = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    f1_bin = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    # Class 0 metrics (Class 0 = Not Paraphrase)
    prec_c0 = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    rec_c0 = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    f1_c0 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # Macro-averaged metrics
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    # Weighted metrics
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # ROC-AUC if probabilities provided
    roc_auc = None
    if y_probs is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_probs))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        digits=4,
        zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "binary_metrics": {
            "precision": round(prec_bin, 4),
            "recall": round(rec_bin, 4),
            "f1_score": round(f1_bin, 4)
        },
        "per_class_metrics": {
            "not_paraphrase": {
                "precision": round(prec_c0, 4),
                "recall": round(rec_c0, 4),
                "f1_score": round(f1_c0, 4)
            },
            "paraphrase": {
                "precision": round(prec_bin, 4),
                "recall": round(rec_bin, 4),
                "f1_score": round(f1_bin, 4)
            }
        },
        "macro_metrics": {
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1_score": round(f1_macro, 4)
        },
        "weighted_f1": round(f1_weighted, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "class_distributions": {
            "test_ground_truth": {
                "not_paraphrase": int((y_true_arr == 0).sum()),
                "paraphrase": int((y_true_arr == 1).sum())
            },
            "predicted_classes": {
                "not_paraphrase": int((y_pred_arr == 0).sum()),
                "paraphrase": int((y_pred_arr == 1).sum())
            }
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        },
        "detailed_class_breakdown": report_dict,
        "classification_report_text": report_text
    }
    return metrics


def plot_confusion_matrix(
    cm_matrix: np.ndarray,
    output_path: str = "results/confusion_matrix.png",
    class_names: Optional[List[str]] = None
) -> None:
    """
    Generate and save a publication-quality confusion matrix heatmap.
    """
    if class_names is None:
        class_names = [ID2LABEL[0], ID2LABEL[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_title("DistilBERT Paraphrase Confusion Matrix", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_curves(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    output_path: str = "results/training_curves.png"
) -> None:
    """
    Generate and save training and validation loss curves across epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    ax.plot(epochs, train_losses, "o-", label="Training Loss", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, val_losses, "s--", label="Validation Loss", color="#ff7f0e", linewidth=2.0)

    ax.set_title("DistilBERT Fine-Tuning Loss Trajectory", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_tfidf_baseline(
    train_pairs: List[Tuple[str, str]],
    train_labels: List[int],
    test_pairs: List[Tuple[str, str]],
    test_labels: List[int]
) -> Dict[str, Any]:
    """
    Train and evaluate a TF-IDF + Logistic Regression baseline on the exact same dataset splits.
    Feature representation: absolute difference and element-wise product of TF-IDF vectors
    for each text pair, representing surface lexical alignment.
    """
    # Combine sentences for unified vocabulary fitting
    corpus = [s for pair in train_pairs for s in pair]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    vectorizer.fit(corpus)

    def extract_pair_features(pairs: List[Tuple[str, str]]) -> np.ndarray:
        feats = []
        for a, b in pairs:
            v_a = vectorizer.transform([a]).toarray()[0]
            v_b = vectorizer.transform([b]).toarray()[0]
            # Difference and product features captures lexical similarity
            diff = np.abs(v_a - v_b)
            prod = v_a * v_b
            cosine = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-9)
            feat_vec = np.hstack([diff, prod, [cosine]])
            feats.append(feat_vec)
        return np.vstack(feats)

    X_train = extract_pair_features(train_pairs)
    X_test = extract_pair_features(test_pairs)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    y_test_arr = np.array(test_labels)

    acc = float(accuracy_score(y_test_arr, y_pred))
    prec = float(precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0))

    return {
        "baseline_name": "TF-IDF + Logistic Regression",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }


def evaluate_bertscore(
    candidates: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Compute supplementary BERTScore (P, R, F1) for a sample of text pairs.
    Handles library availability gracefully.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            candidates,
            references,
            model_type=model_type,
            lang="en",
            verbose=False,
            batch_size=batch_size
        )
        return {
            "status": "success",
            "model_type": model_type,
            "sample_size": len(candidates),
            "mean_precision": round(float(P.mean()), 4),
            "mean_recall": round(float(R.mean()), 4),
            "mean_f1": round(float(F1.mean()), 4)
        }
    except ImportError:
        return {
            "status": "not_installed",
            "message": "bert-score library not installed. Install via 'pip install bert-score' to execute."
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


def save_evaluation_artifacts(
    metrics_data: Dict[str, Any],
    report_text: str,
    confusion_mat: np.ndarray,
    training_history: Optional[Dict[str, List]] = None,
    output_dir: str = "results"
) -> Dict[str, str]:
    """
    Save all empirical evaluation outputs into the results/ directory.
    Returns mapping of artifact names to relative paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # 1. metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    saved_files["metrics_json"] = metrics_path

    # 2. classification_report.txt
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    saved_files["classification_report_txt"] = report_path

    # 3. confusion_matrix.png
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(confusion_mat, output_path=cm_path)
    saved_files["confusion_matrix_png"] = cm_path

    # 4. training_curves.png (if history provided)
    if training_history and "epochs" in training_history:
        curves_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(
            epochs=training_history["epochs"],
            train_losses=training_history["train_losses"],
            val_losses=training_history["val_losses"],
            output_path=curves_path
        )
        saved_files["training_curves_png"] = curves_path

    return saved_files
