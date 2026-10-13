# AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-PAWS%20(labeled__final)-green.svg)](https://huggingface.co/datasets/paws)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A complete, reproducible college NLP mini-project for detecting semantic paraphrasing and surface text overlap using **DistilBERT** fine-tuned on the **PAWS (Paraphrase Adversaries from Word Scrambling)** dataset.

---

## 📌 Project Overview & Key Features

Traditional plagiarism checkers evaluate surface keyword overlap, character n-grams, and TF-IDF similarity. Consequently, they are deceived by:
1. **Adversarial Word Scrambling:** Two sentences with identical vocabulary that express completely different meanings (*"The dog bit the man"* vs. *"The man bit the dog"*).
2. **Lexical Paraphrasing:** Sentences expressing the exact same idea using synonyms and passive voice without sharing distinctive keywords.

This system solves both challenges by combining **deep contextual transformer attention** with **classical surface-level metrics**:
- **Fine-Tuned DistilBERT:** 6-layer sequence classification transformer trained on PAWS sentence pairs to capture syntactic dependencies and semantic roles.
- **Surface Lexical Analysis:** Independent TF-IDF cosine similarity, token Jaccard similarity, and contiguous matching phrase extractor.
- **Five Mandatory Classical NLP Experiments:** Standalone implementations of Sentence Segmentation, Word Tokenization, POS-Guided Lemmatization, Syntactic POS Tagging, and WordNet Semantic Taxonomy Querying.
- **Empirical Evaluation Pipeline:** Held-out test evaluation against a TF-IDF + Logistic Regression baseline, confusion matrix visualization, loss curves, and supplementary BERTScore.
- **Interactive Streamlit Web Dashboard:** Modular, 5-section interactive application with graceful handling of missing model checkpoints.
- **Academic Integrity Boundaries:** Paraphrase classification is clearly presented as a linguistic diagnostic signal, not legal proof of plagiarism or source authorship.

---

## 📂 Project Structure

```
NLP-mpr/
├── app.py                                   # Master 5-section Streamlit Web Application
├── train.py                                 # Standalone PyTorch/Transformers training CLI script
├── requirements.txt                         # Project dependencies
├── README.md                                # Master documentation & execution instructions
├── .gitignore                               # Git ignore configuration
├── notebooks/
│   ├── plagiarism_paraphrase_finetuning.ipynb  # 24-step Google Colab fine-tuning notebook
│   └── generate_notebook.py                 # Reproducible notebook generator script
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── preprocessing.py                     # Text normalization, schema checking, paired tokenization
│   ├── predict.py                           # Checkpoint loading, hardware detection, inference pipeline
│   ├── similarity.py                        # TF-IDF cosine, Jaccard overlap, n-gram extraction
│   └── evaluation.py                        # Classification metrics, confusion matrix, baseline & BERTScore
├── experiments/
│   ├── __init__.py                          # Package initialization
│   └── five_nlp_experiments.py              # Five standalone NLTK classical NLP experiments
├── docs/
│   ├── project_report.md                    # 29-section formal academic project report
│   ├── presentation_slides.md               # 16 presentation slides with speaker notes & 5-min demo
│   ├── fine_tuning_flowchart.md             # Mermaid fine-tuning & evaluation architecture flowchart
│   └── demo_checklist.md                    # 15-step physical evaluation & viva defense checklist
├── models/
│   └── README.md                            # Checkpoint format and Colab import instructions
└── results/
    └── README.md                            # Evaluation artifact specifications (metrics.json, plots)
```

---

## 🛠️ Local Environment Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Create and Activate Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the System

### Option A: Interactive Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. 

*Note: If model weights are not yet downloaded into `models/fine_tuned_distilbert/`, the application will display an informative banner while allowing you to run the lexical similarity analyzer, matching phrase extraction, and all five classical NLP experiments!*

### Option B: Execute the Five Classical NLP Experiments via CLI
Run the standalone NLTK experiments:
```bash
python experiments/five_nlp_experiments.py
```

### Option C: Fine-Tune Locally via CLI
Train DistilBERT directly on your local CPU or Apple Silicon MPS:
```bash
python train.py --subset_train 4000 --subset_val 500 --subset_test 500 --epochs 2 --batch_size 16
```
The trained checkpoint will be saved to `models/fine_tuned_distilbert/` and evaluation artifacts to `results/`.

---

## ☁️ Google Colab GPU Fine-Tuning Guide

For fast model training on a free NVIDIA T4 GPU, use the master Colab notebook:

### 1. Open in Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `notebooks/plagiarism_paraphrase_finetuning.ipynb`.

### 2. Select GPU Runtime
1. In the Colab menu, click **Runtime** $\to$ **Change runtime type**.
2. Select **T4 GPU** under Hardware accelerator.
3. Click **Save**.

### 3. Run All Cells
1. Click **Runtime** $\to$ **Run all** (or run cells sequentially from Step 1 through Step 24).
2. The notebook executes:
   - Dependency installation & GPU verification
   - PAWS dataset loading and zero-leakage split validation
   - Demonstration of the five classical NLP experiments
   - DistilBERT paired sequence tokenization `[CLS] s1 [SEP] s2 [SEP]`
   - Supervised model fine-tuning with AdamW and linear warmup
   - Generation of training loss curve (`training_curves.png`)
   - Evaluation on held-out test data (Accuracy, Precision, Recall, F1)
   - Confusion matrix generation (`confusion_matrix.png`)
   - TF-IDF baseline evaluation on identical test split
   - Supplementary BERTScore evaluation
   - Serialization to `metrics.json` and `classification_report.txt`
   - Packaging into `fine_tuned_distilbert.zip` and `results_artifacts.zip`

### 4. Import Trained Checkpoint to Local Workspace
1. Download `fine_tuned_distilbert.zip` and `results_artifacts.zip` from Colab.
2. In your local project repository, extract the model checkpoint:
   ```bash
   unzip fine_tuned_distilbert.zip -d models/fine_tuned_distilbert/
   ```
3. Extract the evaluation artifacts:
   ```bash
   unzip results_artifacts.zip -d results/
   ```
4. Verify that `models/fine_tuned_distilbert/config.json` exists.
5. Launch or refresh Streamlit (`streamlit run app.py`). The system will automatically detect the checkpoint and enable real-time DistilBERT inference!

---

## 📊 Dataset & Evaluation Benchmarks

- **Dataset:** PAWS (`labeled_final`) from Hugging Face Datasets.
- **Classes:**
  - `0`: **Not Paraphrase** (Adversarial word-scrambled pairs with high lexical overlap but altered meaning).
  - `1`: **Paraphrase** (True semantic equivalence).
- **Zero-Leakage Guarantee:** `src/preprocessing.py` verifies zero string overlap between train and test splits.
- **Baselines:** Direct comparison against TF-IDF + Logistic Regression evaluated on the identical test split.

---

## 📸 Capturing Screenshots for Viva & Report

To capture screenshots for your evaluation viva:
1. **Section A (Overview):** Launch `streamlit run app.py` and capture the title and architecture diagram.
2. **Section B (Comparison - Paraphrase):** Load *Example 1* and capture the high confidence Paraphrase badge and class probabilities.
3. **Section B (Comparison - Adversarial Scramble):** Load *Example 2* and capture the crucial case: TF-IDF shows >85% overlap, while DistilBERT correctly predicts **Not Paraphrase**!
4. **Section C (5 Experiments):** Capture tabs for Exp 1 (Sentence Boundary), Exp 3 (Lemmatization), and Exp 5 (WordNet Semantics).
5. **Section D (Evaluation):** Capture the Confusion Matrix heatmap, Training Loss Curve, and TF-IDF baseline comparative table.

---

## ⚠️ Academic Disclaimer

> **Important:** Paraphrase detection is a diagnostic linguistic analysis that measures semantic equivalence. It does **not** constitute forensic or legal proof of plagiarism. Plagiarism verification requires establishing source provenance, chronological priority, author intent, and citation status—none of which can be determined solely by comparing two isolated text passages.

---

## 📜 License & Acknowledgments
Developed for academic mini-project coursework. Built using Hugging Face Transformers, PyTorch, NLTK, and scikit-learn. PAWS dataset courtesy of Zhang, Baldridge, and He (Google AI, 2019).
