# Physical Demonstration & Viva Defense Checklist

Use this checklist during your college project evaluation, physical demonstration, and faculty viva defense.

## Group Member Participation & Role Division

> **IMPORTANT:** In college project examinations, every team member must actively participate and speak during both the slides and the live software demonstration. Do not let a single person monopolize the presentation.

### Recommended Team Roles
- **Speaker 1 (Problem & NLP Foundations):** Introduces the team, explains the problem statement, defines the difference between lexical similarity and semantic paraphrase detection, and executes the Five Classical NLP Experiments in Streamlit Section C.
- **Speaker 2 (Architecture & Fine-Tuning Pipeline):** Walks through the PAWS dataset, explains DistilBERT architecture, demonstrates the Google Colab fine-tuning notebook (`notebooks/plagiarism_paraphrase_finetuning.ipynb`) and actual training loss curves.
- **Speaker 3 (Live Application & Comparison):** Operates the Streamlit app (Section B). Inputs both standard paraphrases and adversarial scrambled sentences. Demonstrates the live predictions, probabilities, and exact overlapping phrases.
- **Speaker 4 (Evaluation, Metrics & Defense):** Explains Section D (Accuracy, Precision, Recall, F1, Confusion Matrix, TF-IDF baseline advantage, BERTScore), discusses false positives/negatives, and fields faculty viva questions.

---

## 15-Step Physical Demonstration Checklist

Follow these exact steps sequentially when presenting to the evaluator:

- [ ] **Step 1: Explain Problem Statement & Input/Output**
  - Clarify the core limitation of keyword matching: fooled by word scrambling and synonym replacement.
  - Present expected inputs (Passage A, Passage B) and multi-faceted outputs (class label, probabilities, TF-IDF score, matching phrases).
  - Explicitly state that paraphrase classification is an analytical signal, not forensic proof of plagiarism.

- [ ] **Step 2: Show the Dataset & Labels**
  - Open `notebooks/plagiarism_paraphrase_finetuning.ipynb` (or presentation slide 4).
  - Show PAWS (`labeled_final`) schema: `sentence1`, `sentence2`, `label` (`0 = Not Paraphrase`, `1 = Paraphrase`).
  - Explain why PAWS is uniquely adversarial: word scrambling creates high lexical overlap between opposing sentences.

- [ ] **Step 3: Show the Fine-Tuning Notebook**
  - Open `notebooks/plagiarism_paraphrase_finetuning.ipynb` in Google Colab or JupyterLab.
  - Show the 24 sequential cells matching the prompt specification.
  - Highlight device detection (`T4 GPU`), paired sequence encoding `[CLS] A [SEP] B [SEP]`, and zero-leakage split verification.

- [ ] **Step 4: Show Actual Training Logs & Saved Checkpoint**
  - Show the training loop output: epoch loss reduction, validation loss, and total training duration.
  - Open the file manager and show `models/fine_tuned_distilbert/` containing `config.json`, `tokenizer.json`, and weight files.

- [ ] **Step 5: Demonstrate Sentence Boundary Identification (Exp 1)**
  - In Streamlit, navigate to **Section C: Five NLP Experiments** -> **Exp 1: Sentence Boundary**.
  - Click **Run Sentence Segmentation**. Show how Punkt correctly distinguishes abbreviation periods (e.g., `Dr.`) from sentence terminals.

- [ ] **Step 6: Demonstrate Word Tokenization (Exp 2)**
  - Open **Exp 2: Tokenization**.
  - Show whole-word and punctuation tokenization on a test sentence.

- [ ] **Step 7: Demonstrate Lemmatization (Exp 3)**
  - Open **Exp 3: Lemmatization**.
  - Show POS-aware reduction of inflections (e.g., `investigating` $\to$ `investigate`, `better` $\to$ `good`).

- [ ] **Step 8: Demonstrate Syntactic Analysis / POS Tagging (Exp 4)**
  - Open **Exp 4: POS Tagging**.
  - Show Penn Treebank tag assignment and category summaries (nouns, verbs, adjectives, adverbs).

- [ ] **Step 9: Demonstrate WordNet Semantic Analysis (Exp 5)**
  - Open **Exp 5: WordNet Semantics**.
  - Query a word (e.g., `plagiarism` or `paraphrase`). Show synsets, definitions, synonyms, and hypernym trees.

- [ ] **Step 10: Compare Two Text Passages in the Application**
  - In Streamlit, switch to **Section B: Text Comparison**.
  - Test Case A (Standard Paraphrase):
    - *Text A:* `"The company announced a new product yesterday."`
    - *Text B:* `"Yesterday, the company revealed its latest product."`
  - Click **Compare Passages**.

- [ ] **Step 11: Show Actual Model Predictions & Probabilities**
  - Point out the predicted class (`Paraphrase`), confidence percentage, and class probabilities bar.
  - Point out the TF-IDF cosine score and exact matching phrases.

- [ ] **Step 12: Demonstrate the Adversarial Scramble Case (The Crucial Test)**
  - Select *Example 2: PAWS Adversarial Scramble*:
    - *Text A:* `"The police officer arrested the notorious criminal outside the bank."`
    - *Text B:* `"The notorious criminal arrested the police officer outside the bank."`
  - Hit **Compare Passages**.
  - Highlight that TF-IDF reports **over 85% similarity**, but fine-tuned DistilBERT correctly identifies **Not Paraphrase** because it understands semantic roles!

- [ ] **Step 13: Show Accuracy, Precision, Recall, F1, & Confusion Matrix**
  - Navigate to **Section D: Model Evaluation**.
  - Present the quantitative metric cards: Accuracy, Precision, Recall, F1, and Macro F1.
  - Walk through the Confusion Matrix heatmap (`results/confusion_matrix.png`), explaining True Negatives and True Positives.
  - Display the full classification report and training curves.

- [ ] **Step 14: Compare with the TF-IDF Baseline & BERTScore**
  - Present the comparative table in Section D showing that fine-tuned DistilBERT outperforms the TF-IDF + Logistic Regression baseline on the adversarial test split.
  - Present the supplementary BERTScore results on contextual token alignment.

- [ ] **Step 15: Explain Limitations & Future Scope**
  - Explain sequence length limits (128 tokens) and local document pair scope.
  - Discuss future extensions: Longformer for multi-page manuscripts, vector database indexing (Chroma/FAISS) for corpus-scale plagiarism scanning.
  - Invite questions from faculty evaluators.
