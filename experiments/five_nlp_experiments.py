"""
experiments/five_nlp_experiments.py

Implementation of the Five Mandatory NLP Preprocessing Experiments:
1. Sentence Boundary Identification
2. Word Tokenization
3. POS-Aware Lemmatization
4. Syntactic Analysis / Part-of-Speech (POS) Tagging
5. Semantic Analysis via WordNet (Synsets, Synonyms, Hypernyms)

These experiments serve as pedagogical benchmarks in classical NLP,
demonstrated independently of DistilBERT's subword WordPiece tokenizer.
"""

import sys
from typing import Dict, List, Optional, Tuple, Any

# Ensure NLTK data dependencies are gracefully resolved
def ensure_nltk_resources() -> Dict[str, bool]:
    """
    Download required NLTK resources quietly.
    Returns status mapping for each resource.
    """
    import os
    import nltk
    workspace_nltk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
    if os.path.exists(workspace_nltk) and workspace_nltk not in nltk.data.path:
        nltk.data.path.insert(0, workspace_nltk)
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    status = {}
    for path, pkg in resources:
        try:
            nltk.data.find(path)
            status[pkg] = True
        except (LookupError, AttributeError):
            try:
                nltk.download(pkg, quiet=True)
                status[pkg] = True
            except Exception:
                status[pkg] = False
    return status


# ============================================================================
# EXPERIMENT 1: SENTENCE BOUNDARY IDENTIFICATION
# ============================================================================
def run_experiment_1_sentence_boundary(paragraph: Optional[str] = None) -> Dict[str, Any]:
    """
    Experiment 1: Sentence Boundary Identification
    Splits an unstructured text passage into individual grammatical sentences
    using Punkt sentence tokenizer.
    """
    default_text = (
        "Natural language processing enables machines to understand human communication. "
        "Dr. Smith published a benchmark study on plagiarism detection in 2024! "
        "Did you review the experimental findings? They demonstrate why lexical matching fails on adversarial paraphrases."
    )
    input_text = paragraph.strip() if paragraph and paragraph.strip() else default_text

    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        ensure_nltk_resources()
        sentences = sent_tokenize(input_text)
    except Exception:
        # Fallback regular expression sentence splitter
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", input_text) if s.strip()]

    explanation = (
        "Sentence boundary identification (sentence segmentation) parses unstructured discourse "
        "into discrete syntactic units. Unsupervised models such as Punkt distinguish abbreviation periods "
        "(e.g., 'Dr.', 'U.S.') from true sentence-final punctuation marks based on collocation statistics."
    )

    return {
        "experiment_id": 1,
        "title": "Sentence Boundary Identification",
        "input": input_text,
        "operation": "Punkt Sentence Boundary Disambiguation",
        "output": sentences,
        "sentence_count": len(sentences),
        "explanation": explanation
    }


# ============================================================================
# EXPERIMENT 2: TOKENIZATION
# ============================================================================
def run_experiment_2_tokenization(sentence: Optional[str] = None) -> Dict[str, Any]:
    """
    Experiment 2: Word Tokenization
    Deconstructs a sentence into discrete lexical tokens (words and punctuation).
    """
    default_text = "The quick brown fox jumps over the lazy dog's back, effortlessly."
    input_text = sentence.strip() if sentence and sentence.strip() else default_text

    try:
        import nltk
        from nltk.tokenize import word_tokenize
        ensure_nltk_resources()
        tokens = word_tokenize(input_text)
    except Exception:
        import re
        tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", input_text)

    explanation = (
        "Word tokenization isolates atomic lexical constituents while appropriately segmenting "
        "clitics, contractions (e.g., 'dog' + \"'s\"), and terminal punctuation. In contrast to modern subword "
        "tokenizers (such as DistilBERT's WordPiece), classical word tokenizers operate at whole-word boundaries."
    )

    return {
        "experiment_id": 2,
        "title": "Word Tokenization",
        "input": input_text,
        "operation": "Penn Treebank Word Tokenization",
        "output": tokens,
        "token_count": len(tokens),
        "explanation": explanation
    }


# ============================================================================
# EXPERIMENT 3: LEMMATIZATION
# ============================================================================
def get_wordnet_pos(treebank_tag: str):
    """Map POS tag to first character used by WordNetLemmatizer."""
    from nltk.corpus import wordnet
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def run_experiment_3_lemmatization(sentence: Optional[str] = None) -> Dict[str, Any]:
    """
    Experiment 3: POS-Aware Lemmatization
    Reduces inflected and derived morphological variants to their canonical dictionary headword (lemma)
    using WordNet and grammatical part-of-speech context.
    """
    default_text = "The detectives were investigating several running cars and studying better alternatives."
    input_text = sentence.strip() if sentence and sentence.strip() else default_text

    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        ensure_nltk_resources()

        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(input_text)
        tagged = pos_tag(tokens)

        results = []
        for word, tag in tagged:
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            results.append({
                "word": word,
                "pos_tag": tag,
                "lemma": lemma,
                "changed": word.lower() != lemma
            })
    except Exception as e:
        results = [{"word": w, "pos_tag": "UNK", "lemma": w.lower(), "changed": False} for w in input_text.split()]

    explanation = (
        "Lemmatization leverages morphological vocabularies and grammatical context to strip inflectional "
        "suffixes, returning the true base lemma (e.g., 'investigating' -> 'investigate', 'better' -> 'good'). "
        "Unlike crude rule-based stemming (which chops suffixes heuristically), POS-aware lemmatization ensures "
        "valid lexicon membership."
    )

    return {
        "experiment_id": 3,
        "title": "POS-Aware Lemmatization",
        "input": input_text,
        "operation": "WordNet Morphological Analysis with POS Guidance",
        "output": results,
        "explanation": explanation
    }


# ============================================================================
# EXPERIMENT 4: SYNTACTIC ANALYSIS / POS TAGGING
# ============================================================================
TAG_DESCRIPTIONS = {
    "NN": "Noun (singular)", "NNS": "Noun (plural)", "NNP": "Proper noun (singular)", "NNPS": "Proper noun (plural)",
    "VB": "Verb (base)", "VBD": "Verb (past)", "VBG": "Verb (gerund/present participle)",
    "VBN": "Verb (past participle)", "VBP": "Verb (present, non-3d)", "VBZ": "Verb (present, 3d)",
    "JJ": "Adjective", "JJR": "Adjective (comparative)", "JJS": "Adjective (superlative)",
    "RB": "Adverb", "RBR": "Adverb (comparative)", "RBS": "Adverb (superlative)",
    "IN": "Preposition / Conjunction", "DT": "Determiner", "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun", "MD": "Modal auxiliary", "CD": "Cardinal number"
}

def run_experiment_4_pos_tagging(sentence: Optional[str] = None) -> Dict[str, Any]:
    """
    Experiment 4: Syntactic Analysis / Part-of-Speech Tagging
    Assigns Penn Treebank grammatical category tags to each token in a sentence.
    """
    default_text = "The brilliant student solved complex algorithmic problems remarkably fast."
    input_text = sentence.strip() if sentence and sentence.strip() else default_text

    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        ensure_nltk_resources()

        tokens = word_tokenize(input_text)
        tagged = pos_tag(tokens)

        tagged_output = []
        counts = {"nouns": 0, "verbs": 0, "adjectives": 0, "adverbs": 0, "other": 0}

        for token, tag in tagged:
            desc = TAG_DESCRIPTIONS.get(tag, f"Other ({tag})")
            if tag.startswith("NN"):
                counts["nouns"] += 1
            elif tag.startswith("VB"):
                counts["verbs"] += 1
            elif tag.startswith("JJ"):
                counts["adjectives"] += 1
            elif tag.startswith("RB"):
                counts["adverbs"] += 1
            else:
                counts["other"] += 1

            tagged_output.append({
                "token": token,
                "tag": tag,
                "description": desc
            })
    except Exception:
        tagged_output = [{"token": w, "tag": "UNK", "description": "Unknown"} for w in input_text.split()]
        counts = {"nouns": 0, "verbs": 0, "adjectives": 0, "adverbs": 0, "other": len(tagged_output)}

    explanation = (
        "Part-of-Speech (POS) tagging disambiguates syntactic roles based on lexical frequency and "
        "surrounding sequential context. Recognizing nouns, verbs, adjectives, and adverbs enables parsers to "
        "model phrase structure and syntactic dependencies."
    )

    return {
        "experiment_id": 4,
        "title": "Syntactic Analysis / Part-of-Speech Tagging",
        "input": input_text,
        "operation": "Averaged Perceptron Penn Treebank POS Tagger",
        "output": tagged_output,
        "category_summary": counts,
        "explanation": explanation
    }


# ============================================================================
# EXPERIMENT 5: SEMANTIC ANALYSIS
# ============================================================================
def run_experiment_5_semantic_analysis(word: Optional[str] = None) -> Dict[str, Any]:
    """
    Experiment 5: Semantic Analysis using WordNet
    Retrieves synsets, definitions, synonym sets, and hypernyms (is-a hierarchies)
    for a query term, handling missing entries gracefully.
    """
    default_word = "plagiarism"
    query_word = word.strip().lower() if word and word.strip() else default_word

    synset_data = []
    synonyms_set = set()

    try:
        import nltk
        from nltk.corpus import wordnet
        ensure_nltk_resources()

        synsets = wordnet.synsets(query_word)

        for s in synsets:
            # Synonyms within this synset
            lemmas = [l.name().replace("_", " ") for l in s.lemmas()]
            synonyms_set.update(lemmas)

            # Hypernyms (broader conceptual classes)
            hypernym_names = [h.name().split(".")[0].replace("_", " ") for h in s.hypernyms()]

            synset_data.append({
                "synset_id": s.name(),
                "pos": s.pos(),
                "definition": s.definition(),
                "examples": s.examples(),
                "synonyms": lemmas,
                "hypernyms": hypernym_names
            })

        has_synsets = len(synsets) > 0
    except Exception as e:
        synset_data = []
        has_synsets = False

    explanation = (
        "WordNet models lexical semantics through cognitive synonym sets (synsets) interconnected by "
        "conceptual-semantic relations including hypernymy (generalization) and hyponymy (specialization). "
        "This structural ontology enables lexical similarity measurement and taxonomy traversal."
    )

    return {
        "experiment_id": 5,
        "title": "Semantic Analysis via WordNet",
        "input_word": query_word,
        "operation": "WordNet Lexical Database Query",
        "found_synsets": has_synsets,
        "synset_count": len(synset_data),
        "synsets": synset_data,
        "all_synonyms": sorted(list(synonyms_set)),
        "explanation": explanation
    }


def run_all_experiments() -> Dict[str, Any]:
    """Execute all five experiments sequentially with default inputs."""
    return {
        "exp1": run_experiment_1_sentence_boundary(),
        "exp2": run_experiment_2_tokenization(),
        "exp3": run_experiment_3_lemmatization(),
        "exp4": run_experiment_4_pos_tagging(),
        "exp5": run_experiment_5_semantic_analysis()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("FIVE MANDATORY NLP PREPROCESSING EXPERIMENTS (NLTK BENCHMARKS)")
    print("=" * 80)

    # 1. Sentence Boundary
    e1 = run_experiment_1_sentence_boundary()
    print(f"\n[EXPERIMENT 1: {e1['title'].upper()}]")
    print(f"Input: {e1['input']}")
    print("Output Sentences:")
    for idx, s in enumerate(e1["output"], 1):
        print(f"  {idx}. {s}")
    print(f"Note: {e1['explanation']}\n")

    # 2. Tokenization
    e2 = run_experiment_2_tokenization()
    print(f"[EXPERIMENT 2: {e2['title'].upper()}]")
    print(f"Input: {e2['input']}")
    print(f"Tokens ({e2['token_count']}): {e2['output']}")
    print(f"Note: {e2['explanation']}\n")

    # 3. Lemmatization
    e3 = run_experiment_3_lemmatization()
    print(f"[EXPERIMENT 3: {e3['title'].upper()}]")
    print(f"Input: {e3['input']}")
    print("Lemmas:")
    for item in e3["output"]:
        if item["changed"]:
            print(f"  * {item['word']} ({item['pos_tag']}) -> {item['lemma']} [INFLECTION REDUCED]")
        else:
            print(f"    {item['word']} ({item['pos_tag']}) -> {item['lemma']}")
    print(f"Note: {e3['explanation']}\n")

    # 4. POS Tagging
    e4 = run_experiment_4_pos_tagging()
    print(f"[EXPERIMENT 4: {e4['title'].upper()}]")
    print(f"Input: {e4['input']}")
    print("Grammatical Tags:")
    for item in e4["output"]:
        print(f"  {item['token']:<15} | {item['tag']:<6} | {item['description']}")
    print(f"Summary: {e4['category_summary']}")
    print(f"Note: {e4['explanation']}\n")

    # 5. Semantic Analysis
    e5 = run_experiment_5_semantic_analysis("plagiarism")
    print(f"[EXPERIMENT 5: {e5['title'].upper()}]")
    print(f"Word: '{e5['input_word']}' (Found: {e5['synset_count']} synsets)")
    for s in e5["synsets"]:
        print(f"  Synset: {s['synset_id']} | Definition: {s['definition']}")
        print(f"    Synonyms: {s['synonyms']}")
        print(f"    Hypernyms: {s['hypernyms']}")
    print(f"Note: {e5['explanation']}\n")
    print("=" * 80)
    print("ALL FIVE EXPERIMENTS EXECUTED SUCCESSFULLY.")
    print("=" * 80)
