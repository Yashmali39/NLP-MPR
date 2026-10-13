"""
app.py

AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT
Master Streamlit Web Application

Organized into five mandatory sections:
- Section A: Project Overview & Objectives
- Section B: Text Comparison (DistilBERT vs. TF-IDF + Matching Phrases)
- Section C: Five Mandatory Classical NLP Preprocessing Experiments
- Section D: Empirical Model Evaluation & Artifacts
- Section E: Architecture & Model Information
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Configure page metadata
st.set_page_config(
    page_title="AI Plagiarism & Paraphrase Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
from src.preprocessing import clean_text, validate_input_pair, ID2LABEL
from src.similarity import (
    compute_tfidf_cosine_similarity,
    compute_jaccard_similarity,
    extract_overlapping_phrases,
    explain_lexical_vs_semantic,
)
from src.predict import (
    is_checkpoint_available,
    load_fine_tuned_model,
    predict_paraphrase,
    get_optimal_device,
    ModelNotFoundError,
)
from experiments.five_nlp_experiments import (
    run_experiment_1_sentence_boundary,
    run_experiment_2_tokenization,
    run_experiment_3_lemmatization,
    run_experiment_4_pos_tagging,
    run_experiment_5_semantic_analysis,
)

# Cached resource loading for model and tokenizer to prevent reload lag
@st.cache_resource(show_spinner="Loading Fine-Tuned DistilBERT Checkpoint...")
def get_cached_model():
    if not is_checkpoint_available("models/fine_tuned_distilbert"):
        return None, None, None
    try:
        model, tokenizer, device = load_fine_tuned_model("models/fine_tuned_distilbert")
        return model, tokenizer, device
    except Exception as e:
        return None, None, None


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & INTEGRITY NOTICE
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Navigation")
active_section = st.sidebar.radio(
    "Select Project Section:",
    [
        "Section A: Project Overview",
        "Section B: Text Comparison",
        "Section C: NLP Analysis Lab",
        "Section D: Model Evaluation",
        "Section E: Model Information"
    ],
    index=1  # Default to Text Comparison for live demonstrations
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
checkpoint_exists = is_checkpoint_available("models/fine_tuned_distilbert")
if checkpoint_exists:
    st.sidebar.success("✅ Fine-Tuned DistilBERT Checkpoint Loaded")
else:
    st.sidebar.warning("⚠️ No Fine-Tuned Checkpoint in `models/fine_tuned_distilbert/`")
    st.sidebar.caption("Run Colab notebook or `train.py` to produce weights.")

device_detected = get_optimal_device()
st.sidebar.info(f"⚡ Device: **{device_detected.type.upper()}**")

st.sidebar.markdown("---")
st.sidebar.caption("NLP Mini-Project • Academic Year 2026")


# =============================================================================
# SECTION A: PROJECT OVERVIEW
# =============================================================================
if active_section == "Section A: Project Overview":
    st.title("AI-Powered Plagiarism and Paraphrase Detection System")
    st.subheader("Fine-Tuned DistilBERT Architecture for Contextual Semantic Analysis")

    st.markdown("""
    ### 1. Abstract & System Description
    This project addresses the challenging NLP task of detecting semantic paraphrasing and academic text overlap.
    Traditional plagiarism detectors rely heavily on lexical surface matching (such as string overlap, n-gram hashing, 
    and TF-IDF cosine similarity). These methods are readily deceived by **adversarial word scrambling**, 
    synonym substitutions, and passive-to-active voice grammatical reformulations.

    Our system fine-tunes **DistilBERT** (`distilbert/distilbert-base-uncased`) on the challenging 
    **PAWS (Paraphrase Adversaries from Word Scrambling)** dataset. DistilBERT processes text pairs natively 
    through cross-attention, capturing contextual semantics and syntactic role assignments to accurately 
    discriminate between true paraphrases and scrambled non-paraphrases.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        #### 🎯 Core Objectives
        - **Supervised Fine-Tuning:** Fine-tune DistilBERT sequence classifier on PAWS text pairs.
        - **Dual-Perspective Analysis:** Deliver both deep semantic probability and surface lexical metrics.
        - **Classical NLP Foundations:** Implement and demonstrate five classical NLTK preprocessing steps.
        - **Empirical Rigor:** Evaluate on held-out test splits, comparing against a TF-IDF baseline.
        """)

    with col2:
        st.warning("""
        #### ⚠️ Scope & Important Limitations
        - **Paraphrase $\\neq$ Proof of Plagiarism:** Paraphrase detection is a diagnostic linguistic signal. 
          It does not determine authorship, chronology, or illicit copying.
        - **Isolated Comparison:** Evaluates two user-submitted passages; it does not scrape or index the web.
        - **Length Constraint:** Optimized for sequence passages up to 128 tokens per pair.
        """)

    st.markdown("### How the System Works")
    st.markdown("""
    ```
    ┌────────────────────────┐      ┌────────────────────────┐
    │     Passage A Text     │      │     Passage B Text     │
    └───────────┬────────────┘      └───────────┬────────────┘
                │                               │
                ▼                               ▼
    ┌────────────────────────────────────────────────────────┐
    │                   Input Preprocessing                  │
    │  - Unicode Normalization   - Length & Whitespace Check │
    └───────────────┬───────────────────────────┬────────────┘
                    │                           │
                    ▼                           ▼
        [Surface Lexical Analysis]      [Deep Contextual Analysis]
        - TF-IDF Cosine Similarity       - [CLS] A [SEP] B [SEP]
        - Contiguous N-Gram Matching     - Cross-Attention Layers
        - Jaccard Token Overlap          - Softmax Class Probabilities
                    │                           │
                    └───────────────┬───────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │  Consolidated Comparative Report & Academic Note  │
        └───────────────────────────────────────────────────┘
    ```
    """)


# =============================================================================
# SECTION B: TEXT COMPARISON
# =============================================================================
elif active_section == "Section B: Text Comparison":
    st.title("⚖️ Text Comparison: Semantic & Lexical Analysis")
    st.markdown(
        "Compare two passages to predict whether they represent semantic paraphrases or distinct statements, "
        "alongside surface-level lexical similarity and exact matching phrase extraction."
    )

    # Preset sample selector for rapid evaluator demonstration
    st.subheader("1. Select or Enter Text Passages")
    preset_choice = st.selectbox(
        "Load an Illustrative Benchmark Example (Optional):",
        [
            "Custom Input",
            "Example 1: True Semantic Paraphrase (Vocabulary & Word Swap)",
            "Example 2: PAWS Adversarial Scramble (High Word Overlap, Different Meaning)",
            "Example 3: Completely Distinct Passages (Different Topics)"
        ]
    )

    default_a = "The company announced a new product yesterday."
    default_b = "Yesterday, the company revealed its latest product."

    if preset_choice == "Example 1: True Semantic Paraphrase (Vocabulary & Word Swap)":
        default_a = "The company announced a new product yesterday."
        default_b = "Yesterday, the company revealed its latest product."
    elif preset_choice == "Example 2: PAWS Adversarial Scramble (High Word Overlap, Different Meaning)":
        default_a = "The police officer arrested the notorious criminal outside the bank."
        default_b = "The notorious criminal arrested the police officer outside the bank."
    elif preset_choice == "Example 3: Completely Distinct Passages (Different Topics)":
        default_a = "Quantum computing relies on qubits to perform exponential calculations."
        default_b = "Photosynthesis allows green plants to convert solar radiation into chemical energy."

    col_a, col_b = st.columns(2)
    with col_a:
        text_a_input = st.text_area(
            "Text A (Original or Reference Passage):",
            value=default_a,
            height=140
        )
    with col_b:
        text_b_input = st.text_area(
            "Text B (Suspected Paraphrase / Comparison Passage):",
            value=default_b,
            height=140
        )

    compare_clicked = st.button("🔍 Compare Passages", type="primary", use_container_width=True)

    if compare_clicked:
        is_valid, err_msg = validate_input_pair(text_a_input, text_b_input)
        if not is_valid:
            st.error(f"Input Validation Error: {err_msg}")
        else:
            clean_a = clean_text(text_a_input)
            clean_b = clean_text(text_b_input)

            # 1. Lexical Similarity Calculation
            tfidf_score = compute_tfidf_cosine_similarity(clean_a, clean_b)
            jaccard_score = compute_jaccard_similarity(clean_a, clean_b)
            overlapping_phrases = extract_overlapping_phrases(clean_a, clean_b)

            # 2. Semantic DistilBERT Inference
            model, tokenizer, device = get_cached_model()
            has_model = (model is not None and tokenizer is not None)

            st.markdown("---")
            st.subheader("2. Comparison Results")

            # Layout: Left column = Semantic DistilBERT, Right column = Lexical Metrics
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("#### 🤖 Fine-Tuned DistilBERT (Semantic)")
                if has_model:
                    pred_res = predict_paraphrase(model, tokenizer, clean_a, clean_b, device=device)
                    pred_label = pred_res["predicted_label"]
                    prob_para = pred_res["probabilities"]["Paraphrase"]
                    prob_not_para = pred_res["probabilities"]["Not Paraphrase"]

                    if pred_label == "Paraphrase":
                        st.success(f"### Predicted Class: **{pred_label}**")
                    else:
                        st.info(f"### Predicted Class: **{pred_label}**")

                    st.markdown(f"**Confidence:** `{pred_res['confidence'] * 100:.2f}%` | **Latency:** `{pred_res['latency_ms']} ms`")

                    # Probabilities Breakdown
                    st.write("**Model Class Probabilities:**")
                    st.progress(float(prob_para), text=f"Paraphrase: {prob_para * 100:.2f}%")
                    st.progress(float(prob_not_para), text=f"Not Paraphrase: {prob_not_para * 100:.2f}%")
                else:
                    st.warning("⚠️ **Fine-Tuned Checkpoint Not Found**")
                    st.markdown("""
                    In adherence to project integrity standards, an untrained base model is **never** silently substituted.
                    
                    **To enable DistilBERT predictions:**
                    1. Open and execute `notebooks/plagiarism_paraphrase_finetuning.ipynb` in Google Colab (GPU accelerated).
                    2. Download `fine_tuned_distilbert.zip` and extract to `models/fine_tuned_distilbert/`.
                    3. Or execute local training via CLI: `python train.py --subset_train 2000`.
                    4. Refresh this page once `models/fine_tuned_distilbert/config.json` is present.
                    """)
                    pred_label = "Unavailable (Model Not Fine-Tuned)"
                    prob_para = 0.0

            with res_col2:
                st.markdown("#### 📏 Surface Lexical Similarity")
                st.metric(
                    label="TF-IDF Cosine Similarity",
                    value=f"{tfidf_score * 100:.2f}%"
                )
                st.metric(
                    label="Jaccard Token Overlap",
                    value=f"{jaccard_score * 100:.2f}%"
                )

                if overlapping_phrases:
                    st.write(f"**Exact Overlapping Phrases ({len(overlapping_phrases)} detected):**")
                    for p in overlapping_phrases[:5]:
                        st.markdown(f"- `\"{p['phrase']}\"` *({p['word_count']} words)*")
                else:
                    st.caption("No multi-word exact contiguous phrase matches detected.")

            # 3. Pedagogical Explanation of Divergence / Convergence
            st.markdown("---")
            st.subheader("3. Linguistic Interpretation: Lexical vs. Semantic")
            explanation_data = explain_lexical_vs_semantic(tfidf_score, pred_label, prob_para)

            st.info(f"**Surface Evaluation:** {explanation_data['lexical_assessment']}")
            st.markdown(f"**Relationship Analysis:** {explanation_data['relationship_explanation']}")

            # Mandatory Plagiarism Disclaimer
            st.warning(f"**Academic Notice:** {explanation_data['plagiarism_disclaimer']}")


# =============================================================================
# SECTION C: FIVE NLP EXPERIMENTS
# =============================================================================
elif active_section == "Section C: NLP Analysis Lab":
    st.title("🧪 Five Mandatory Classical NLP Preprocessing Experiments")
    st.markdown(
        "Demonstration of foundational NLP operations using NLTK. "
        "These experiments serve as pedagogical benchmarks in classical computational linguistics, "
        "independent of DistilBERT's deep contextual tokenization."
    )

    exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5 = st.tabs([
        "Exp 1: Sentence Boundary",
        "Exp 2: Tokenization",
        "Exp 3: Lemmatization",
        "Exp 4: POS Tagging",
        "Exp 5: WordNet Semantics"
    ])

    # Experiment 1
    with exp_tab1:
        st.subheader("Experiment 1: Sentence Boundary Identification")
        st.markdown("Splits an unstructured discourse passage into individual sentences using Punkt sentence tokenizer.")
        exp1_input = st.text_area(
            "Input Paragraph:",
            value=(
                "Natural language processing enables machines to understand human communication. "
                "Dr. Smith published a benchmark study on plagiarism detection in 2024! "
                "Did you review the experimental findings? They demonstrate why lexical matching fails on adversarial paraphrases."
            ),
            key="exp1_input"
        )
        if st.button("Run Sentence Segmentation", key="exp1_btn"):
            res1 = run_experiment_1_sentence_boundary(exp1_input)
            st.write(f"**Operation Performed:** `{res1['operation']}`")
            st.write(f"**Detected Sentences ({res1['sentence_count']}):**")
            for idx, s in enumerate(res1["output"], 1):
                st.markdown(f"**[{idx}]** {s}")
            st.info(f"**Theoretical Note:** {res1['explanation']}")

    # Experiment 2
    with exp_tab2:
        st.subheader("Experiment 2: Word Tokenization")
        st.markdown("Deconstructs a sentence into discrete lexical tokens (words and punctuation).")
        exp2_input = st.text_input(
            "Input Sentence:",
            value="The quick brown fox jumps over the lazy dog's back, effortlessly.",
            key="exp2_input"
        )
        if st.button("Run Word Tokenization", key="exp2_btn"):
            res2 = run_experiment_2_tokenization(exp2_input)
            st.write(f"**Operation Performed:** `{res2['operation']}`")
            st.write(f"**Token Count:** `{res2['token_count']}`")
            st.write("**Extracted Tokens:**")
            st.code(str(res2["output"]))
            st.info(f"**Theoretical Note:** {res2['explanation']}")

    # Experiment 3
    with exp_tab3:
        st.subheader("Experiment 3: POS-Aware Lemmatization")
        st.markdown("Reduces inflected words to their canonical dictionary headword (lemma) using WordNet and POS context.")
        exp3_input = st.text_input(
            "Input Sentence:",
            value="The detectives were investigating several running cars and studying better alternatives.",
            key="exp3_input"
        )
        if st.button("Run Lemmatization", key="exp3_btn"):
            res3 = run_experiment_3_lemmatization(exp3_input)
            st.write(f"**Operation Performed:** `{res3['operation']}`")
            lemma_df = pd.DataFrame(res3["output"])
            st.dataframe(lemma_df, use_container_width=True)
            st.info(f"**Theoretical Note:** {res3['explanation']}")

    # Experiment 4
    with exp_tab4:
        st.subheader("Experiment 4: Syntactic Analysis / Part-of-Speech Tagging")
        st.markdown("Assigns Penn Treebank grammatical category tags (nouns, verbs, adjectives, adverbs).")
        exp4_input = st.text_input(
            "Input Sentence:",
            value="The brilliant student solved complex algorithmic problems remarkably fast.",
            key="exp4_input"
        )
        if st.button("Run POS Tagging", key="exp4_btn"):
            res4 = run_experiment_4_pos_tagging(exp4_input)
            st.write(f"**Operation Performed:** `{res4['operation']}`")

            # Show category counts in columns
            c_cols = st.columns(4)
            c_cols[0].metric("Nouns", res4["category_summary"]["nouns"])
            c_cols[1].metric("Verbs", res4["category_summary"]["verbs"])
            c_cols[2].metric("Adjectives", res4["category_summary"]["adjectives"])
            c_cols[3].metric("Adverbs", res4["category_summary"]["adverbs"])

            pos_df = pd.DataFrame(res4["output"])
            st.dataframe(pos_df, use_container_width=True)
            st.info(f"**Theoretical Note:** {res4['explanation']}")

    # Experiment 5
    with exp_tab5:
        st.subheader("Experiment 5: WordNet Semantic Analysis")
        st.markdown("Retrieves synsets, definitions, synonym sets, and hypernyms (is-a hierarchies) for a target word.")
        exp5_input = st.text_input(
            "Query Word:",
            value="plagiarism",
            key="exp5_input"
        )
        if st.button("Search WordNet Semantics", key="exp5_btn"):
            res5 = run_experiment_5_semantic_analysis(exp5_input)
            st.write(f"**Operation Performed:** `{res5['operation']}`")
            if res5["found_synsets"]:
                st.write(f"**Synsets Found:** `{res5['synset_count']}`")
                st.write(f"**All Synonyms Identified:** `{res5['all_synonyms']}`")
                for s in res5["synsets"]:
                    with st.expander(f"Synset: `{s['synset_id']}` (POS: {s['pos']})"):
                        st.markdown(f"**Definition:** {s['definition']}")
                        st.markdown(f"**Synonyms:** {', '.join(s['synonyms'])}")
                        st.markdown(f"**Hypernyms (Generalizations):** {', '.join(s['hypernyms'])}")
                        if s["examples"]:
                            st.markdown(f"**Examples:** {s['examples']}")
            else:
                st.warning(f"No synsets found in WordNet for '{res5['input_word']}'.")
            st.info(f"**Theoretical Note:** {res5['explanation']}")


# =============================================================================
# SECTION D: MODEL EVALUATION
# =============================================================================
elif active_section == "Section D: Model Evaluation":
    st.title("📊 Empirical Model Evaluation & Test Results")
    st.markdown(
        "Quantitative performance metrics computed on the held-out PAWS test split. "
        "All figures reflect actual execution results without fabrication."
    )

    metrics_file = "results/metrics.json"
    report_file = "results/classification_report.txt"
    cm_file = "results/confusion_matrix.png"
    curves_file = "results/training_curves.png"

    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        db_eval = metrics_data.get("distilbert_evaluation", {})
        train_meta = metrics_data.get("training_metadata", {})
        baseline_eval = metrics_data.get("tfidf_baseline_comparison", {})
        bertscore_eval = metrics_data.get("supplementary_bertscore", {})

        # Top metric cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Test Accuracy", f"{db_eval.get('accuracy', 0.0) * 100:.2f}%")
        m_col2.metric("Precision (Paraphrase)", f"{db_eval.get('binary_precision', 0.0) * 100:.2f}%")
        m_col3.metric("Recall (Paraphrase)", f"{db_eval.get('binary_recall', 0.0) * 100:.2f}%")
        m_col4.metric("F1-Score (Paraphrase)", f"{db_eval.get('binary_f1', 0.0) * 100:.2f}%")
        m_col5.metric("Macro F1", f"{db_eval.get('macro_f1', 0.0) * 100:.2f}%")

        st.markdown("---")
        # Visualizations row
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("Confusion Matrix")
            if os.path.exists(cm_file):
                st.image(cm_file, use_container_width=True)
            else:
                st.caption("Confusion matrix image not found.")

        with v_col2:
            st.subheader("Training & Validation Loss Trajectory")
            if os.path.exists(curves_file):
                st.image(curves_file, use_container_width=True)
            else:
                st.caption("Training curve image not found.")

        # Baseline Comparison & BERTScore
        st.markdown("---")
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.subheader("Baseline Comparison: TF-IDF vs. DistilBERT")
            if baseline_eval:
                b_df = pd.DataFrame([
                    {
                        "Model Architecture": "TF-IDF + Logistic Regression",
                        "Accuracy": f"{baseline_eval.get('accuracy', 0.0) * 100:.2f}%",
                        "F1-Score": f"{baseline_eval.get('f1_score', 0.0) * 100:.2f}%"
                    },
                    {
                        "Model Architecture": "Fine-Tuned DistilBERT (Ours)",
                        "Accuracy": f"{db_eval.get('accuracy', 0.0) * 100:.2f}%",
                        "F1-Score": f"{db_eval.get('binary_f1', 0.0) * 100:.2f}%"
                    }
                ])
                st.dataframe(b_df, use_container_width=True)
                acc_diff = (db_eval.get('accuracy', 0.0) - baseline_eval.get('accuracy', 0.0)) * 100
                st.success(f"**Empirical Advantage:** Fine-Tuned DistilBERT outperforms the surface baseline by **{acc_diff:+.2f} percentage points** on the PAWS adversarial dataset.")
            else:
                st.caption("TF-IDF baseline comparison not yet generated.")

        with comp_col2:
            st.subheader("Supplementary Semantic Metric: BERTScore")
            if bertscore_eval and bertscore_eval.get("status") == "success":
                st.write(f"Evaluated on {bertscore_eval.get('sample_size')} sample pairs:")
                st.write(f"- **Mean BERTScore Precision:** `{bertscore_eval.get('mean_precision')}`")
                st.write(f"- **Mean BERTScore Recall:** `{bertscore_eval.get('mean_recall')}`")
                st.write(f"- **Mean BERTScore F1:** `{bertscore_eval.get('mean_f1')}`")
                st.caption("Note: BERTScore evaluates contextual token alignment, complementing binary classification accuracy.")
            else:
                st.info("BERTScore calculation: Can be computed in Colab via Step 20 or CLI.")

        # Classification Report
        st.markdown("---")
        st.subheader("Full Classification Report")
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.caption("Classification report file not found.")

    else:
        st.warning("⚠️ **Evaluation Artifacts Not Yet Generated Locally**")
        st.markdown("""
        In accordance with project guidelines, evaluation metrics are **never fabricated** or pre-filled with synthetic numbers.
        
        **How to Generate Real Evaluation Artifacts:**
        1. **Option A (Google Colab - Recommended for Speed):**
           - Open `notebooks/plagiarism_paraphrase_finetuning.ipynb`.
           - Run cells 1 through 24 using a free T4 GPU.
           - Extract `results_artifacts.zip` into `results/` in this repository.
        2. **Option B (Local CLI Execution):**
           - Execute: `python train.py --subset_train 4000 --subset_val 500 --subset_test 500`
        3. Refresh this page to view your live, empirical metrics and charts.
        """)


# =============================================================================
# SECTION E: MODEL INFORMATION
# =============================================================================
elif active_section == "Section E: Model Information":
    st.title("ℹ️ Model Architecture & Technical Specifications")

    st.markdown("""
    ### Technical Specification Sheet
    | Property | Specification / Implementation Detail |
    | :--- | :--- |
    | **Base Transformer Model** | `distilbert/distilbert-base-uncased` |
    | **Original Architecture** | 6 Transformer encoder layers, 12 attention heads, 768 hidden dimensions |
    | **Parameter Count** | ~66.3 Million parameters (40% smaller and 60% faster than BERT-base) |
    | **Distillation Technique** | Triple loss: soft target distillation, hard cross-entropy, cosine embedding |
    | **Input Format** | Native paired sequence: `[CLS] text_a [SEP] text_b [SEP]` |
    | **Max Sequence Length** | 128 tokens |
    | **Dataset** | PAWS (`labeled_final`) — 49,401 train / 8,000 val / 8,000 test |
    | **Classes** | `0: Not Paraphrase`, `1: Paraphrase` |
    | **Target Adversarial Property** | Word-scrambled pairs sharing high lexical overlap with opposing meanings |
    | **Optimization** | AdamW ($lr=2\\times 10^{-5}$, weight decay=0.01) with linear warmup schedule |
    """)

    st.subheader("Technology Stack & Roles")
    tech_data = [
        {"Library / Framework": "PyTorch", "Role": "Deep learning tensor operations, autograd, and training loop", "Used": "Yes"},
        {"Library / Framework": "Hugging Face Transformers", "Role": "DistilBERT architecture, tokenization, and pretrained weights", "Used": "Yes"},
        {"Library / Framework": "Hugging Face Datasets", "Role": "Loading and streaming official PAWS labeled_final splits", "Used": "Yes"},
        {"Library / Framework": "scikit-learn", "Role": "Evaluation metrics (F1, precision, recall, confusion matrix) & TF-IDF baseline", "Used": "Yes"},
        {"Library / Framework": "NLTK", "Role": "Five mandatory preprocessing experiments (sentence splitting, POS, WordNet)", "Used": "Yes"},
        {"Library / Framework": "Streamlit", "Role": "Interactive user application and evaluator demonstration interface", "Used": "Yes"},
        {"Library / Framework": "BERTScore", "Role": "Supplementary semantic token embedding alignment evaluation", "Used": "Yes"},
        {"Library / Framework": "Google Cloud Natural Language API", "Role": "Cloud NLP alternative (documented, not implemented)", "Used": "No (Local Model Prioritized)"},
        {"Library / Framework": "AWS Comprehend", "Role": "Cloud NLP alternative (documented, not implemented)", "Used": "No (Local Model Prioritized)"},
    ]
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True)

    st.subheader("Checkpoint Directory Inspection")
    target_dir = "models/fine_tuned_distilbert"
    if os.path.exists(target_dir):
        files = os.listdir(target_dir)
        st.success(f"Checkpoint directory `{target_dir}` found with files: `{files}`")
    else:
        st.info(f"Checkpoint directory `{target_dir}` does not yet exist. Ready for Colab import.")
