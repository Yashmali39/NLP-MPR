"""
src/preprocessing.py

Data cleaning, input validation, tokenization preprocessing, and PAWS dataset
schema handling for DistilBERT fine-tuning and inference.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any

# Canonical label mapping across dataset, model, and application
ID2LABEL: Dict[int, str] = {
    0: "Not Paraphrase",
    1: "Paraphrase"
}

LABEL2ID: Dict[str, int] = {
    "Not Paraphrase": 0,
    "Paraphrase": 1
}


def clean_text(text: str) -> str:
    """
    Standardize raw user or dataset text.
    - Applies Unicode NFKC normalization.
    - Strips non-printable and control characters while preserving valid punctuation.
    - Collapses contiguous whitespace into a single space.
    - Strips leading and trailing whitespace.
    """
    if text is None:
        return ""
    text = str(text)
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline and tab
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    # Normalize excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Strip whitespace
    return text.strip()


def validate_input_pair(text_a: str, text_b: str, min_words: int = 1) -> Tuple[bool, Optional[str]]:
    """
    Validate that two input passages are non-empty and satisfy minimum length constraints.
    Returns (is_valid, error_message).
    """
    clean_a = clean_text(text_a)
    clean_b = clean_text(text_b)

    if not clean_a:
        return False, "Text A is empty or contains only whitespace/control characters."
    if not clean_b:
        return False, "Text B is empty or contains only whitespace/control characters."

    words_a = clean_a.split()
    words_b = clean_b.split()

    if len(words_a) < min_words:
        return False, f"Text A must contain at least {min_words} word(s)."
    if len(words_b) < min_words:
        return False, f"Text B must contain at least {min_words} word(s)."

    return True, None


def tokenize_paired_text(
    tokenizer: Any,
    text_a: str,
    text_b: str,
    max_length: int = 128,
    return_tensors: Optional[str] = "pt"
) -> Dict[str, Any]:
    """
    Tokenize a pair of texts using DistilBERT's paired sequence formatting:
    [CLS] text_a [SEP] text_b [SEP]
    Note: DistilBERT does not utilize token_type_ids; it relies strictly on
    input_ids and attention_mask.
    """
    clean_a = clean_text(text_a)
    clean_b = clean_text(text_b)

    encoded = tokenizer(
        clean_a,
        clean_b,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors
    )
    return encoded


def validate_paws_records(df: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Inspect, validate, and clean PAWS dataset records.
    - Confirms presence of 'sentence1', 'sentence2', and 'label'.
    - Drops null / NaN entries.
    - Removes empty or whitespace-only records.
    - Validates binary labels (0 or 1).
    - Checks for internal duplicate pairs.
    Returns:
        cleaned_df: pandas DataFrame containing valid rows
        stats: dictionary describing cleaning statistics
    """
    import pandas as pd
    required_cols = ["sentence1", "sentence2", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"PAWS dataset missing required column '{col}'. Present: {list(df.columns)}")

    initial_count = len(df)
    df_clean = df.copy()

    # Drop nulls
    df_clean = df_clean.dropna(subset=required_cols)
    null_dropped = initial_count - len(df_clean)

    # Clean text columns
    df_clean["sentence1"] = df_clean["sentence1"].astype(str).apply(clean_text)
    df_clean["sentence2"] = df_clean["sentence2"].astype(str).apply(clean_text)

    # Filter out empty strings
    df_clean = df_clean[(df_clean["sentence1"] != "") & (df_clean["sentence2"] != "")]
    empty_dropped = (initial_count - null_dropped) - len(df_clean)

    # Validate label is 0 or 1
    df_clean["label"] = pd.to_numeric(df_clean["label"], errors="coerce")
    df_clean = df_clean.dropna(subset=["label"])
    df_clean["label"] = df_clean["label"].astype(int)
    df_clean = df_clean[df_clean["label"].isin([0, 1])]

    # Deduplicate pair collisions
    dedup_subset = ["sentence1", "sentence2"]
    duplicates_count = df_clean.duplicated(subset=dedup_subset).sum()
    df_clean = df_clean.drop_duplicates(subset=dedup_subset).reset_index(drop=True)

    class_counts = df_clean["label"].value_counts().to_dict()
    stats = {
        "initial_rows": initial_count,
        "valid_rows": len(df_clean),
        "null_rows_dropped": null_dropped,
        "empty_rows_dropped": empty_dropped,
        "duplicates_dropped": int(duplicates_count),
        "class_distribution": {
            ID2LABEL.get(k, str(k)): int(v) for k, v in class_counts.items()
        }
    }
    return df_clean, stats


def check_split_leakage(train_df: Any, test_df: Any) -> Dict[str, Any]:
    """
    Check for text-pair leakage between train and test splits to guarantee
    academic rigor and prevent inflated evaluation metrics.
    """
    train_pairs = set(zip(train_df["sentence1"].str.lower(), train_df["sentence2"].str.lower()))
    test_pairs = set(zip(test_df["sentence1"].str.lower(), test_df["sentence2"].str.lower()))

    # Exact pair overlap
    overlap = train_pairs.intersection(test_pairs)

    # Also check inverted pairs (sentence2, sentence1)
    train_pairs_rev = set(zip(train_df["sentence2"].str.lower(), train_df["sentence1"].str.lower()))
    rev_overlap = train_pairs_rev.intersection(test_pairs)

    return {
        "exact_overlap_count": len(overlap),
        "inverted_overlap_count": len(rev_overlap),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "leakage_detected": len(overlap) > 0 or len(rev_overlap) > 0
    }
