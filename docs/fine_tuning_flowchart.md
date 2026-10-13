# DistilBERT Fine-Tuning and Evaluation Flowchart

This document illustrates the end-to-end engineering pipeline for the **AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT**.

```mermaid
flowchart TD
    subgraph DataPipeline ["1. Data Acquisition & Validation"]
        A["Hugging Face PAWS Dataset<br/>(labeled_final configuration)"] --> B["Data Inspection & Schema Validation<br/>(sentence1, sentence2, label)"]
        B --> C["Data Cleaning & Hygiene<br/>(Drop Nulls, Strip Control Chars, Deduplicate)"]
        C --> D["Split Verification & Zero-Leakage Audit<br/>(Train: 49.4k | Val: 8k | Test: 8k)"]
    end

    subgraph Preprocessing ["2. Tokenization & Model Setup"]
        D --> E["Active Subset Selection<br/>(e.g., 8,000 Train / 1,000 Val / 1,000 Test)"]
        E --> F["Load Pretrained DistilBERT Tokenizer<br/>(distilbert/distilbert-base-uncased)"]
        F --> G["Paired Sequence Tokenization<br/>[CLS] sentence1 [SEP] sentence2 [SEP]<br/>(max_length=128, attention_mask)"]
        G --> H["PyTorch TextPairDataset & DataLoaders<br/>(Batch Size: 16 Train / 32 Eval)"]
    end

    subgraph FineTuning ["3. Model Fine-Tuning Loop"]
        H --> I["Initialize DistilBERT Sequence Classifier<br/>(num_labels=2, id2label, label2id)"]
        I --> J["Optimizer & Schedule Configuration<br/>(AdamW lr=2e-5, Weight Decay=0.01, Linear Warmup)"]
        J --> K["Epoch Training Loop (CUDA / MPS / CPU)<br/>Forward Pass -> Cross-Entropy Loss -> Backward -> Clip"]
        K --> L["Validation Monitoring<br/>(Compute Validation Loss & Accuracy per Epoch)"]
        L --> M{"Training Complete?<br/>(Target Epochs Reached)"}
        M -- No --> K
        M -- Yes --> N["Save Fine-Tuned Checkpoint<br/>(models/fine_tuned_distilbert/)"]
    end

    subgraph Evaluation ["4. Held-Out Evaluation & Baselines"]
        N --> O["Held-Out Test Set Inference<br/>(Unseen 1,000 Test Pairs)"]
        O --> P["Calculate Quantitative Metrics<br/>(Accuracy, Precision, Recall, Binary & Macro F1)"]
        O --> Q["Generate Confusion Matrix & Classification Report<br/>(results/confusion_matrix.png, results/classification_report.txt)"]
        O --> R["TF-IDF + Logistic Regression Baseline<br/>(Evaluated on Identical Test Split)"]
        O --> S["Supplementary BERTScore Semantic Evaluation<br/>(Contextual Token Alignment P, R, F1)"]
        P & Q & R & S --> T["Serialize Metrics & Loss History<br/>(results/metrics.json)"]
    end

    subgraph Application ["5. Interactive Deployment"]
        N --> U["Streamlit Web Application (app.py)<br/>@st.cache_resource Checkpoint Loader"]
        T --> V["Section D: Empirical Evaluation Dashboard"]
        U --> W["Section B: Real-Time Dual Comparison<br/>- Semantic Probability (DistilBERT)<br/>- Lexical TF-IDF Cosine Similarity<br/>- Contiguous Overlapping Phrases"]
        W --> X["Linguistic Interpretation & Academic Disclaimer"]
    end
```

## Description of Architectural Stages

1. **Data Acquisition & Validation:** Loads the official PAWS `labeled_final` configuration from Hugging Face Datasets, verifies data types, enforces zero null records, normalizes text into Unicode NFKC format, and conducts rigorous cross-split set intersection tests to prevent train-test data leakage.
2. **Tokenization & Model Setup:** Encodes sentence pairs using DistilBERT's paired input format `[CLS] text_a [SEP] text_b [SEP]`. DistilBERT natively learns bidirectional cross-attention across both segments without needing segment token embeddings (`token_type_ids`).
3. **Model Fine-Tuning Loop:** Employs AdamW optimizer with a 10% linear warmup schedule to fine-tune the 66-million parameter transformer. Tracks training and validation loss after every epoch.
4. **Held-Out Evaluation & Baselines:** Computes classification metrics on the official held-out test split. Benchmarks DistilBERT directly against a TF-IDF + Logistic Regression baseline trained on the exact same split, demonstrating transformer superiority on word-scrambled adversarial text pairs.
5. **Interactive Deployment:** Serves the fine-tuned model via a high-performance Streamlit UI (`app.py`), presenting model probabilities, lexical cosine metrics, exact matching phrases, and a clear academic plagiarism disclaimer.
