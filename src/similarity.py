"""
src/similarity.py

Lexical similarity calculation, TF-IDF cosine similarity, n-gram phrase overlap
detection, and comparative analysis between lexical and semantic similarity.
"""

import re
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between Text A and Text B using TF-IDF word vectors.
    Returns a float score in [0.0, 1.0].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        # Cosine similarity between row 0 (text_a) and row 1 (text_b)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except ValueError:
        # Occurs if all tokens are stop words or vocabulary is empty
        # Fall back to character n-gram TF-IDF
        char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = char_vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute word-level Jaccard similarity (token intersection over union).
    """
    words_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    words_b = set(re.findall(r"\b\w+\b", text_b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    return float(len(intersection) / len(union))


def extract_overlapping_phrases(
    text_a: str,
    text_b: str,
    min_phrase_len: int = 3,
    max_phrase_len: int = 8
) -> List[Dict[str, Any]]:
    """
    Extract exact contiguous word sequences (n-grams) of length min_phrase_len to
    max_phrase_len that appear in both Text A and Text B.
    Returns list of dicts with phrase text and word count, sorted by length descending.
    """
    words_a = re.findall(r"\b[\w'-]+\b", text_a.lower())
    words_b = re.findall(r"\b[\w'-]+\b", text_b.lower())

    if len(words_a) < min_phrase_len or len(words_b) < min_phrase_len:
        return []

    def get_ngrams(words: List[str], n: int) -> Set[Tuple[str, ...]]:
        return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

    overlapping_tuples: Set[Tuple[str, ...]] = set()

    for n in range(min_phrase_len, min(max_phrase_len, len(words_a), len(words_b)) + 1):
        ngrams_a = get_ngrams(words_a, n)
        ngrams_b = get_ngrams(words_b, n)
        common = ngrams_a.intersection(ngrams_b)
        overlapping_tuples.update(common)

    # Prune sub-phrases contained in longer matching phrases
    sorted_tuples = sorted(overlapping_tuples, key=lambda t: len(t), reverse=True)
    maximal_phrases: List[str] = []

    for t in sorted_tuples:
        phrase_str = " ".join(t)
        # Check if this phrase is a strict sub-phrase of any already accepted longer phrase
        if not any(phrase_str in longer for longer in maximal_phrases):
            maximal_phrases.append(phrase_str)

    results = [
        {"phrase": p, "word_count": len(p.split())}
        for p in maximal_phrases
    ]
    return results


def explain_lexical_vs_semantic(
    tfidf_score: float,
    semantic_label: str,
    semantic_probability: float
) -> Dict[str, str]:
    """
    Provide an academic, clear comparison explaining the divergence or alignment
    between lexical surface similarity and deep semantic paraphrase detection.
    """
    # Characterize lexical overlap
    if tfidf_score >= 0.70:
        lex_desc = "High surface lexical overlap (large proportion of shared keywords and vocabulary)."
    elif tfidf_score >= 0.35:
        lex_desc = "Moderate lexical overlap (partial sharing of subject terms or phrasing)."
    else:
        lex_desc = "Low lexical overlap (distinct vocabulary, differing terminology, or extensive rewording)."

    # Interpret interaction
    is_paraphrase = (semantic_label == "Paraphrase")

    if is_paraphrase and tfidf_score < 0.40:
        relationship = (
            "DIVERGENCE (High Semantic Paraphrase, Low Lexical Overlap): "
            "The model detected that both texts convey the same core proposition or meaning despite using "
            "completely different vocabulary, synonyms, and passive/active voice restructuring. "
            "Traditional keyword or TF-IDF detectors fail on this case, illustrating the power of deep contextual attention."
        )
    elif not is_paraphrase and tfidf_score > 0.60:
        relationship = (
            "ADVERSARIAL SCRAMBLING (Low Semantic Paraphrase, High Lexical Overlap): "
            "The texts share a high percentage of identical words, but their grammatical ordering or semantic roles "
            "(e.g., subject-object inversion or negation) completely alter the factual meaning. "
            "This is the hallmark of the PAWS dataset, where lexical overlap fooled surface algorithms but DistilBERT correctly identifies differing meaning."
        )
    elif is_paraphrase and tfidf_score >= 0.40:
        relationship = (
            "CONVERGENCE (Paraphrase with Lexical Sharing): "
            "Both surface keyword overlap and deep semantic contextual embeddings agree that the passages "
            "are paraphrased versions of each other."
        )
    else:
        relationship = (
            "CONVERGENCE (Distinct Non-Paraphrase): "
            "Both surface vocabulary and deep semantic analysis confirm that the two passages discuss "
            "unrelated topics or distinct factual propositions."
        )

    disclaimer = (
        "IMPORTANT ACADEMIC DISCLAIMER: Paraphrase detection is a diagnostic linguistic signal, "
        "not legal or academic proof of plagiarism. Plagiarism verification requires establishing source provenance, "
        "chronological priority, lack of citation, and intentionality—none of which can be determined "
        "solely by comparing two isolated text passages."
    )

    return {
        "lexical_assessment": lex_desc,
        "relationship_explanation": relationship,
        "plagiarism_disclaimer": disclaimer
    }
