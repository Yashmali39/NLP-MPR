# AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT
## Academic Presentation Slides & Demonstration Guide

---

### Slide 1: Title & Team Information
- **Project Title:** AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT
- **Course:** Natural Language Processing (NLP) Mini-Project
- **Academic Term:** October 2026
- **Technology Stack:** PyTorch, Hugging Face Transformers, PAWS Dataset, NLTK, Streamlit

**Speaker Notes:**
> "Good morning, respected evaluators. Today, our team presents our NLP mini-project: an AI-powered paraphrase and plagiarism detection system built by fine-tuning DistilBERT on adversarial sentence pairs. We will demonstrate how deep transformer attention overcomes the critical limitations of classical lexical matching."

---

### Slide 2: Problem Statement
- **Core Dilemma:** Traditional plagiarism detection relies primarily on surface string matching, token hashing, and TF-IDF cosine similarity.
- **Vulnerability:** Such systems fail when text undergoes synonym substitution, active/passive voice inversion, or adversarial word scrambling.
- **The Core Challenge:** Distinguishing between sentences that share identical vocabulary but mean completely different things (e.g., *"The dog bit the man"* vs. *"The man bit the dog"*), versus sentences with distinct words that express the same idea.

**Speaker Notes:**
> "The core challenge in modern plagiarism and paraphrase detection is the disconnect between surface lexical overlap and true semantic meaning. Scrambled sentences fool keyword detectors, while extensive rewording evades token overlap checks. Our project addresses this using contextual transformer representation."

---

### Slide 3: Objectives & Project Scope
- **Supervised Transformer Fine-Tuning:** Fine-tune `distilbert-base-uncased` on paired sequence classification.
- **Adversarial Paraphrase Benchmark:** Train on the PAWS dataset, specifically constructed to test sensitivity to word order and syntax.
- **Dual-Perspective Analysis:** Simultaneously deliver deep semantic probability, TF-IDF cosine similarity, and contiguous matching phrases.
- **Classical NLP Foundations:** Implement and demonstrate five classical NLTK preprocessing steps.
- **Academic Boundary:** Paraphrase detection serves as an analytical component; it does not claim to prove legal plagiarism or verify source attribution.

**Speaker Notes:**
> "Our objective is to deliver a robust, fine-tuned transformer classifier alongside classical NLP metrics. We explicitly demarcate the boundary between paraphrase detection and plagiarism proof: our system measures semantic equivalence, which is a diagnostic indicator, not forensic proof of authorship."

---

### Slide 4: Dataset — PAWS (Paraphrase Adversaries from Word Scrambling)
- **Source:** Hugging Face Datasets (`paws`, configuration `labeled_final`).
- **Composition:** 49,401 training pairs, 8,000 validation pairs, 8,000 test pairs.
- **Key Feature:** Pairs generated via language-model word swapping and back-translation, curated with human annotations.
- **Target Challenge:** Pairs have high lexical overlap (average BLEU/word overlap > 0.70) even when the semantic meaning is completely negated or reversed.
- **Classes:** Binary classification (`0: Not Paraphrase`, `1: Paraphrase`).

**Speaker Notes:**
> "We selected the PAWS dataset specifically because standard paraphrase benchmarks like QQP or MRPC contain superficial lexical cues. PAWS text pairs share almost identical vocabulary, forcing the model to learn true syntactic dependencies and semantic roles rather than relying on keyword co-occurrence."

---

### Slide 5: Proposed System Architecture
- **Input Pipeline:** Receives Passage A and Passage B, applies Unicode NFKC normalization, and validates input integrity.
- **Paired-Input Tokenization:** Encodes `[CLS] Passage A [SEP] Passage B [SEP]` up to 128 tokens.
- **Deep Contextual Attention:** 6 DistilBERT transformer encoder layers compute cross-passage attention.
- **Classification Head:** Linear projection on pooled `[CLS]` token with Softmax probability output.
- **Parallel Lexical Analysis:** Independent TF-IDF vectorization and n-gram overlap extraction for comparison.

**Speaker Notes:**
> "Our architecture processes the paired passages jointly through DistilBERT's cross-attention layers. Simultaneously, an independent lexical engine computes TF-IDF similarity and extracts contiguous overlapping phrases. This dual-stream architecture highlights the divergence between surface keywords and semantic intent."

---

### Slide 6: System Architecture Diagram
```
[User Passage A] ────┐
                     ├─► [Preprocessing] ─► [Tokenization [CLS] A [SEP] B [SEP]]
[User Passage B] ────┘          │                          │
                                ▼                          ▼
                        [TF-IDF & N-Grams]        [DistilBERT 6 Encoders]
                                │                          │
                                ▼                          ▼
                         Lexical Overlap %          Softmax Probabilities
                                └──────────────┬───────────┘
                                               ▼
                                  [Streamlit Interface Report]
```

**Speaker Notes:**
> "As visualized here, the input is bifurcated into a surface-level lexical pipeline and our deep transformer neural network. The results converge in the Streamlit application to provide a clear, comprehensive diagnostic report."

---

### Slide 7: Five Mandatory NLP Preprocessing Experiments
1. **Sentence Boundary Identification:** Punkt unsupervised sentence boundary disambiguation.
2. **Word Tokenization:** Penn Treebank word and punctuation segmenter.
3. **POS-Aware Lemmatization:** WordNet lemmatizer guided by Penn Treebank POS tags.
4. **Syntactic POS Tagging:** Averaged Perceptron tagger classifying grammatical categories.
5. **WordNet Semantic Analysis:** Taxonomy querying for synsets, definitions, synonyms, and hypernyms.

**Speaker Notes:**
> "Before deploying our deep learning model, we implemented five foundational NLP experiments using NLTK. These experiments demonstrate how classical rule-based and statistical methods extract morphological, syntactic, and lexical structure from unstructured text."

---

### Slide 8: DistilBERT Architecture Overview
- **Student-Teacher Distillation:** Compressed variant of BERT-base trained via knowledge distillation (Sanh et al., 2019).
- **Architectural Pruning:** 6 transformer encoder layers (vs. 12 in BERT), 768 hidden dimensions, 12 attention heads.
- **Efficiency Profile:** 66.3 million parameters — 40% smaller and 60% faster than BERT-base while retaining 97% of language understanding capability.
- **Memory Optimization:** Enables rapid fine-tuning on consumer hardware and low-latency inference (<20 ms per pair).

**Speaker Notes:**
> "DistilBERT strikes the optimal balance between computational efficiency and semantic accuracy. By retaining 97% of BERT's capabilities with only 66 million parameters, it is exceptionally well-suited for interactive, real-time college NLP systems."

---

### Slide 9: Fine-Tuning Workflow
1. **Tokenizer & Model Initialization:** `distilbert-base-uncased` with binary classification head.
2. **Batch Sequencing:** Dynamically padded batches with attention masks.
3. **Loss Function:** Binary Cross-Entropy with Softmax (`CrossEntropyLoss`).
4. **Optimization:** AdamW optimizer ($lr=2\times 10^{-5}$, weight decay=0.01).
5. **Learning Rate Schedule:** Linear decay with 10% warmup steps.
6. **Validation Checkpoints:** Continuous loss tracking per epoch to avoid overfitting.

**Speaker Notes:**
> "During fine-tuning, all encoder layers and the classification head are jointly updated using AdamW and a linear warmup schedule. This gently adjusts pretrained representations toward the subtle syntactic patterns necessary for PAWS classification."

---

### Slide 10: Implementation & Streamlit Interface
- **Section A:** Project Overview, Objectives, and Scope Limitations.
- **Section B:** Interactive Text Comparison with real-time probabilities, lexical scores, and matching phrases.
- **Section C:** Interactive testing suite for all Five Classical NLP Experiments.
- **Section D:** Empirical Evaluation dashboard showing live metrics, confusion matrix, and loss curves.
- **Section E:** Model technical specifications, hardware detection, and checkpoint status.

**Speaker Notes:**
> "Our Streamlit application provides a clean, modular demonstration platform. It features strict architectural integrity: if the fine-tuned checkpoint is missing, it explicitly informs the user rather than faking predictions with an untrained base model."

---

### Slide 11: Evaluation Methodology
- **Held-Out Test Set:** Evaluated exclusively on unseen test pairs from PAWS.
- **Metrics Tracked:**
  - Accuracy: Overall classification correctness.
  - Precision: Avoidance of false paraphrase alarms.
  - Recall: Sensitivity in catching disguised paraphrases.
  - F1-Score: Harmonic balance between precision and recall (binary & macro).
- **Confusion Matrix:** True Negatives, False Positives, False Negatives, True Positives.
- **Comparative Baselines:** TF-IDF + Logistic Regression on the identical split, plus supplementary BERTScore.

**Speaker Notes:**
> "Our evaluation follows strict machine learning standards: zero data leakage between splits, evaluation exclusively on held-out test data, and direct benchmarking against a competitive TF-IDF baseline."

---

### Slide 12: Experimental Results
*(Refer to `results/metrics.json` generated from live execution)*
- **Model Checkpoint:** `models/fine_tuned_distilbert/`
- **Training Trajectory:** Rapid convergence within 2 epochs; training loss decreases smoothly alongside validation loss.
- **Evaluation Metrics:**
  - Accuracy: High 80s% on the challenging PAWS adversarial split.
  - Paraphrase F1-Score: Balanced performance across both classes.
  - Inference Latency: ~10–25 ms per pair on GPU/MPS.

**Speaker Notes:**
> "The empirical results demonstrate robust convergence. DistilBERT successfully navigates the word-scrambling traps of PAWS, producing strong classification accuracy and balanced F1 scores."

---

### Slide 13: Sample Predictions & Qualitative Analysis
- **Case 1: Standard Paraphrase**
  - *Text A:* "The company announced a new product yesterday."
  - *Text B:* "Yesterday, the company revealed its latest product."
  - *Result:* Paraphrase (High Confidence), Moderate TF-IDF.
- **Case 2: Adversarial Word Scramble (The PAWS Hallmark)**
  - *Text A:* "The dog bit the mailman near the gate."
  - *Text B:* "The mailman bit the dog near the gate."
  - *Result:* Not Paraphrase (High Confidence), High TF-IDF (85%+ overlap).
  - *Significance:* TF-IDF fails completely; DistilBERT correctly identifies semantic divergence.

**Speaker Notes:**
> "These two cases highlight why this project matters. In Case 2, 90% of the words are identical. A traditional lexical system flags this as 90% plagiarized text. Our fine-tuned DistilBERT correctly predicts 'Not Paraphrase' because its cross-attention understands semantic role assignments."

---

### Slide 14: Baseline Comparison: DistilBERT vs. TF-IDF
- **TF-IDF + Logistic Regression Baseline:**
  - Relies on word n-gram frequencies and difference vectors.
  - Severely degraded by PAWS because non-paraphrases have almost identical word counts.
- **Fine-Tuned DistilBERT:**
  - Outperforms the TF-IDF baseline by a substantial margin on the adversarial dataset.
  - Confirms that contextual embeddings are mandatory when word scrambling is present.

**Speaker Notes:**
> "When evaluated on the identical test split, our fine-tuned DistilBERT achieves a marked improvement over the TF-IDF baseline. This confirms our core hypothesis: surface keywords alone cannot solve paraphrase detection under adversarial conditions."

---

### Slide 15: Limitations & Future Scope
- **Current Limitations:**
  - Sequence length capped at 128 tokens (optimized for sentences/short passages).
  - Evaluates isolated pairs; does not crawl databases or web indices.
  - Does not evaluate source attribution or quotation validity.
- **Future Enhancements:**
  - Longformer / BigBird backbones for full-document academic thesis comparison.
  - Cross-lingual paraphrase detection (multilingual DistilBERT / XLM-RoBERTa).
  - Integration with vector databases (ChromaDB / FAISS) for corpus-scale similarity search.

**Speaker Notes:**
> "Looking ahead, extending the context window to support multi-page documents and integrating vector databases for large-scale document corpus retrieval are natural extensions of this work."

---

### Slide 16: Conclusion & Summary
- Successfully built an end-to-end, reproducible NLP system combining classical preprocessing with fine-tuned transformers.
- Fine-tuned DistilBERT on the PAWS dataset, solving the adversarial word-scramble problem.
- Implemented and demonstrated all five classical NLTK experiments.
- Delivered an interactive Streamlit application with dual-perspective reporting.
- Enforced strict academic integrity and transparent evaluation reporting throughout.

**Speaker Notes:**
> "In conclusion, we have designed, fine-tuned, evaluated, and deployed a complete NLP mini-project that satisfies every theoretical and practical requirement. Thank you, and we welcome your questions."

---

## 5-Minute Physical Demonstration Script

| Elapsed Time | Action | Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 0:45** | Introduce Problem & Slides 1–3 | State project title, team members, and the fundamental flaw of keyword plagiarism detection (adversarial word scrambling). |
| **0:45 - 1:30** | Demonstrate Classical NLP (Section C) | Open Streamlit Section C. Run Exp 1 (Sentence splitting), Exp 3 (POS-aware lemmatization), and Exp 5 (WordNet synsets). Explain that these classical foundations are independent of transformer tokenization. |
| **1:30 - 3:00** | Live Text Comparison (Section B) | Select "Example 2: PAWS Adversarial Scramble" (*The police officer arrested the criminal* vs. *The criminal arrested the police officer*). Hit **Compare**. Show that TF-IDF reports 85%+ similarity, but DistilBERT correctly predicts **Not Paraphrase** with high confidence. Show matching phrases. |
| **3:00 - 4:00** | Empirical Evaluation (Section D) | Navigate to Section D. Display the Confusion Matrix, Training Loss Trajectory, Classification Report, and TF-IDF baseline comparison. Emphasize zero data leakage and held-out test evaluation. |
| **4:00 - 4:30** | Technical Specs & Architecture (Section E) | Show Section E: Base model parameter count, PAWS schema, and explain why DistilBERT was selected over heavier models. |
| **4:30 - 5:00** | Conclusion & Q&A | Reiterate that paraphrase classification is a linguistic indicator, not forensic plagiarism proof. Invite questions from evaluators. |
