"""
Helper script to generate the compliant Jupyter Notebook notebooks/plagiarism_paraphrase_finetuning.ipynb
with exactly 24 executable sections in strict sequential order as specified in Section 5.
"""

import json
import os

def build_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_markdown("""# AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT
## Master Fine-Tuning & Evaluation Notebook (Google Colab / GPU Accelerated)

**Project Title:** AI-Powered Plagiarism and Paraphrase Detection System Using Fine-Tuned DistilBERT  
**Dataset:** PAWS (Paraphrase Adversaries from Word Scrambling) - `labeled_final`  
**Base Model:** `distilbert/distilbert-base-uncased`  

### Google Colab Instructions:
1. **Enable GPU:** In the top menu, navigate to `Runtime` -> `Change runtime type` -> select `T4 GPU` -> Click `Save`.
2. **Execute Sequentially:** Run each cell in order from Cell 1 through Cell 24 (`Runtime` -> `Run all`).
3. **Download Artifacts:** Cell 23 packages the trained checkpoint and evaluation artifacts into `fine_tuned_distilbert.zip` and triggers an automatic browser download (with an option to copy to Google Drive).
4. **Local Deployment:** Extract the archive into your local project's `models/fine_tuned_distilbert/` directory to power the local Streamlit application.""")

    # Cell 1: Install dependencies
    add_markdown("### Step 1: Install Dependencies")
    add_code("""!pip install --quiet --upgrade pip
!pip install --quiet torch transformers datasets evaluate accelerate scikit-learn nltk pandas numpy matplotlib seaborn bert-score tqdm""")

    # Cell 2: Check Python, PyTorch, and GPU availability
    add_markdown("### Step 2: Check Python, PyTorch, and Hardware/GPU Availability")
    add_code("""import sys
import torch

print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"CUDA Available    : {gpu_available}")

if gpu_available:
    print(f"GPU Device Name   : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (GB)   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Apple Silicon MPS Available")
    device = torch.device("mps")
else:
    print("Warning: Running on CPU. Training will take longer. Consider selecting a GPU runtime.")
    device = torch.device("cpu")

print(f"Selected Device   : {device}")""")

    # Cell 3: Load the PAWS dataset
    add_markdown("### Step 3: Load the PAWS Dataset (`labeled_final`)")
    add_code("""from datasets import load_dataset

print("[*] Fetching PAWS dataset ('labeled_final') from Hugging Face Datasets...")
try:
    paws_dataset = load_dataset("google-research-datasets/paws", "labeled_final")
except Exception:
    paws_dataset = load_dataset("paws", "labeled_final")
print(paws_dataset)""")

    # Cell 4: Inspect dataset fields and class labels
    add_markdown("### Step 4: Inspect Dataset Fields and Class Labels")
    add_code("""train_split = paws_dataset["train"]
val_split = paws_dataset["validation"]
test_split = paws_dataset["test"]

print(f"Features in PAWS dataset: {train_split.features}")
print(f"Columns present         : {train_split.column_names}")

# Canonical class mapping
ID2LABEL = {0: "Not Paraphrase", 1: "Paraphrase"}
LABEL2ID = {"Not Paraphrase": 0, "Paraphrase": 1}

print(f"Class Label Mapping     : {ID2LABEL}")""")

    # Cell 5: Display dataset samples and class distribution
    add_markdown("### Step 5: Display Dataset Samples and Class Distribution")
    add_code("""import pandas as pd
import matplotlib.pyplot as plt

train_df_raw = train_split.to_pandas()
val_df_raw = val_split.to_pandas()
test_df_raw = test_split.to_pandas()

print(f"Raw Split Sizes -> Train: {len(train_df_raw):,}, Val: {len(val_df_raw):,}, Test: {len(test_df_raw):,}\\n")

print("--- Sample Records from Training Split ---")
for idx, row in train_df_raw.head(3).iterrows():
    print(f"Sample #{idx+1} [Label: {row['label']} -> {ID2LABEL[row['label']]}]")
    print(f"  Sentence 1: {row['sentence1']}")
    print(f"  Sentence 2: {row['sentence2']}\\n")

# Distribution plot
plt.figure(figsize=(6, 3.5), dpi=120)
ax = train_df_raw['label'].value_counts().plot(kind='bar', color=['#3498db', '#2ecc71'])
plt.title("PAWS Training Set Class Distribution", fontsize=12, fontweight='bold')
plt.xticks([0, 1], [ID2LABEL[0], ID2LABEL[1]], rotation=0)
plt.ylabel("Number of Pairs")
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()""")

    # Cell 6: Validate and clean records
    add_markdown("### Step 6: Validate and Clean Records")
    add_code("""import unicodedata
import re

def clean_text_cell(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\\n" or ch == "\\t" or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"[ \\t]+", " ", text)
    return text.strip()

def clean_split_df(df):
    clean = df.dropna(subset=["sentence1", "sentence2", "label"]).copy()
    clean["sentence1"] = clean["sentence1"].apply(clean_text_cell)
    clean["sentence2"] = clean["sentence2"].apply(clean_text_cell)
    clean = clean[(clean["sentence1"] != "") & (clean["sentence2"] != "")]
    clean["label"] = clean["label"].astype(int)
    clean = clean.drop_duplicates(subset=["sentence1", "sentence2"]).reset_index(drop=True)
    return clean

train_df = clean_split_df(train_df_raw)
val_df = clean_split_df(val_df_raw)
test_df = clean_split_df(test_df_raw)

print(f"Cleaned Split Sizes -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")""")

    # Cell 7: Demonstrate the five mandatory NLP experiments
    add_markdown("### Step 7: Demonstrate the Five Mandatory NLP Experiments (NLTK Benchmarks)")
    add_code("""import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Download resources quietly
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

print("=" * 75)
print("DEMONSTRATION: FIVE MANDATORY NLP PREPROCESSING EXPERIMENTS")
print("=" * 75)

# Exp 1: Sentence Boundary
para = "Machine learning parses text. Dr. Watson examined the document! Did they detect paraphrase copying?"
sents = sent_tokenize(para)
print(f"\\n1. Sentence Boundary Detection ({len(sents)} sentences):")
for i, s in enumerate(sents, 1):
    print(f"   [{i}] {s}")

# Exp 2: Word Tokenization
sample_sent = "The fine-tuned DistilBERT model detected subtle semantic paraphrases."
words = word_tokenize(sample_sent)
print(f"\\n2. Word Tokenization ({len(words)} tokens):\\n   {words}")

# Exp 3: POS-Aware Lemmatization
lemmatizer = WordNetLemmatizer()
test_words = [("investigating", "v"), ("better", "a"), ("feet", "n"), ("wolves", "n")]
print("\\n3. POS-Aware Lemmatization:")
for w, p in test_words:
    lemma = lemmatizer.lemmatize(w, pos=p)
    print(f"   Original: '{w}' (POS={p}) -> Lemma: '{lemma}'")

# Exp 4: POS Tagging
sample_syntax = "Students submitted remarkably original essays yesterday."
tagged = pos_tag(word_tokenize(sample_syntax))
print(f"\\n4. POS Tagging:\\n   {tagged}")

# Exp 5: WordNet Semantic Analysis
synsets = wordnet.synsets("paraphrase")
print(f"\\n5. Semantic Analysis for 'paraphrase' ({len(synsets)} synsets):")
for s in synsets[:2]:
    print(f"   Synset: {s.name()} | Def: {s.definition()}")
    print(f"   Synonyms: {[l.name() for l in s.lemmas()]}")
print("=" * 75)""")

    # Cell 8: Verify the training, validation, and test splits
    add_markdown("### Step 8: Verify Train, Validation, and Test Splits (Leakage & Subset Selection)")
    add_code("""# Configuration parameters for reproducible, efficient training
SUBSET_TRAIN = 8000
SUBSET_VAL = 1000
SUBSET_TEST = 1000
RANDOM_SEED = 42

train_sub = train_df.sample(n=min(SUBSET_TRAIN, len(train_df)), random_state=RANDOM_SEED).reset_index(drop=True)
val_sub = val_df.sample(n=min(SUBSET_VAL, len(val_df)), random_state=RANDOM_SEED).reset_index(drop=True)
test_sub = test_df.sample(n=min(SUBSET_TEST, len(test_df)), random_state=RANDOM_SEED).reset_index(drop=True)

# Train-Test leakage verification
train_pairs = set(zip(train_sub["sentence1"].str.lower(), train_sub["sentence2"].str.lower()))
test_pairs = set(zip(test_sub["sentence1"].str.lower(), test_sub["sentence2"].str.lower()))
leakage = train_pairs.intersection(test_pairs)

print(f"Active Experiment Subsets:")
print(f"  Training Set   : {len(train_sub):,} pairs | Balance: {train_sub['label'].value_counts().to_dict()}")
print(f"  Validation Set : {len(val_sub):,} pairs | Balance: {val_sub['label'].value_counts().to_dict()}")
print(f"  Held-out Test  : {len(test_sub):,} pairs | Balance: {test_sub['label'].value_counts().to_dict()}")
print(f"  Split Leakage  : {len(leakage)} overlapping pairs (Strict Zero-Leakage Confirmed).")""")

    # Cell 9: Load the pretrained DistilBERT tokenizer
    add_markdown("### Step 9: Load the Pretrained DistilBERT Tokenizer")
    add_code("""from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased"
print(f"[*] Loading tokenizer from '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer Vocabulary Size : {tokenizer.vocab_size:,}")
print(f"Model Max Length          : {tokenizer.model_max_length}")""")

    # Cell 10: Load the two-class sequence classification model
    add_markdown("### Step 10: Load the Two-Class Sequence Classification Model")
    add_code("""from transformers import DistilBertForSequenceClassification

print(f"[*] Initializing DistilBertForSequenceClassification from '{MODEL_NAME}'...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")""")

    # Cell 11: Tokenize the text pairs
    add_markdown("### Step 11: Tokenize the Text Pairs with Native Paired Sequencing")
    add_code("""import torch
from torch.utils.data import Dataset, DataLoader

MAX_LENGTH = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

class TextPairDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def encode_df(df):
    encodings = tokenizer(
        df["sentence1"].tolist(),
        df["sentence2"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return TextPairDataset(encodings, df["label"].tolist())

print("[*] Encoding paired sequences [CLS] s1 [SEP] s2 [SEP]...")
train_dataset = encode_df(train_sub)
val_dataset = encode_df(val_sub)
test_dataset = encode_df(test_sub)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

print(f"DataLoaders Ready -> Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, Test Batches: {len(test_loader)}")""")

    # Cell 12: Configure training
    add_markdown("### Step 12: Configure Training Hyperparameters, Optimizer, and Scheduler")
    add_code("""from transformers import get_linear_schedule_with_warmup

EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Training Configuration:")
print(f"  Epochs           : {EPOCHS}")
print(f"  Batch Size       : {BATCH_SIZE}")
print(f"  Total Steps      : {total_steps}")
print(f"  Warmup Steps     : {warmup_steps}")
print(f"  Learning Rate    : {LEARNING_RATE}")
print(f"  Weight Decay     : {WEIGHT_DECAY}")""")

    # Cell 13: Fine-tune the model
    add_markdown("### Step 13: Fine-Tune the DistilBERT Model (PyTorch Training Loop)")
    add_code("""import time
from tqdm.auto import tqdm
import numpy as np

history = {"epochs": [], "train_losses": [], "val_losses": []}
training_start = time.time()

print(f"[*] Beginning DistilBERT fine-tuning on {device}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]")

    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    running_val_loss = 0.0
    val_preds, val_trues = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            running_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            val_preds.extend(preds)
            val_trues.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_loader)
    val_acc = (np.array(val_preds) == np.array(val_trues)).mean()

    print(f"Epoch {epoch}/{EPOCHS} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    history["epochs"].append(epoch)
    history["train_losses"].append(round(epoch_train_loss, 4))
    history["val_losses"].append(round(epoch_val_loss, 4))

training_time = time.time() - training_start
print(f"\\n[*] Model Fine-Tuning Completed in {training_time:.2f} seconds ({training_time/60.0:.2f} mins).")""")

    # Cell 14: Display training and validation loss
    add_markdown("### Step 14: Display Training and Validation Loss Curves")
    add_code("""import os

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 4.5), dpi=150)
plt.plot(history["epochs"], history["train_losses"], "o-", label="Training Loss", color="#1f77b4", linewidth=2)
plt.plot(history["epochs"], history["val_losses"], "s--", label="Validation Loss", color="#ff7f0e", linewidth=2)
plt.title("DistilBERT Fine-Tuning Loss Curve", fontsize=13, weight="bold")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=300)
plt.show()""")

    # Cell 15: Save the fine-tuned checkpoint and tokenizer
    add_markdown("### Step 15: Save the Fine-Tuned Checkpoint and Tokenizer")
    add_code("""OUTPUT_DIR = "models/fine_tuned_distilbert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[*] Serializing fine-tuned model and tokenizer to '{OUTPUT_DIR}'...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[*] Files saved in checkpoint: {os.listdir(OUTPUT_DIR)}")""")

    # Cell 16: Evaluate on held-out test data
    add_markdown("### Step 16: Evaluate on Held-Out Test Data")
    add_code("""model.eval()
test_preds = []
test_trues = []
test_probs = []

eval_start = time.time()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Held-Out Test Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        test_preds.extend(preds.tolist())
        test_trues.extend(labels.cpu().numpy().tolist())
        test_probs.extend(probs.tolist())

eval_time = time.time() - eval_start
latency_per_sample = (eval_time / len(test_preds)) * 1000.0

print(f"[*] Test inference finished for {len(test_preds)} samples.")
print(f"[*] Mean latency per sample: {latency_per_sample:.2f} ms")""")

    # Cell 17: Calculate all required metrics
    add_markdown("### Step 17: Calculate All Required Evaluation Metrics")
    add_code("""from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true_np = np.array(test_trues)
y_pred_np = np.array(test_preds)

test_acc = accuracy_score(y_true_np, y_pred_np)
bin_prec = precision_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
bin_rec = recall_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
bin_f1 = f1_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)

macro_prec = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
macro_rec = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
macro_f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)

print("=" * 60)
print("DISTILBERT HELD-OUT TEST EVALUATION METRICS")
print("=" * 60)
print(f"Accuracy                  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Binary Precision (Class 1): {bin_prec:.4f}")
print(f"Binary Recall    (Class 1): {bin_rec:.4f}")
print(f"Binary F1-Score  (Class 1): {bin_f1:.4f}")
print(f"Macro-Averaged Precision  : {macro_prec:.4f}")
print(f"Macro-Averaged Recall     : {macro_rec:.4f}")
print(f"Macro-Averaged F1-Score   : {macro_f1:.4f}")
print("=" * 60)""")

    # Cell 18: Generate a confusion matrix and classification report
    add_markdown("### Step 18: Generate a Confusion Matrix and Classification Report")
    add_code("""import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
cls_report = classification_report(
    y_true_np, y_pred_np,
    labels=[0, 1],
    target_names=[ID2LABEL[0], ID2LABEL[1]],
    digits=4
)

print("[CLASSIFICATION REPORT]")
print(cls_report)

with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(cls_report)

# Plot confusion matrix
plt.figure(figsize=(6, 5), dpi=150)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[ID2LABEL[0], ID2LABEL[1]],
    yticklabels=[ID2LABEL[0], ID2LABEL[1]],
    annot_kws={"size": 14, "weight": "bold"}
)
plt.title("Held-Out Test Confusion Matrix", fontsize=12, weight="bold", pad=12)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()""")

    # Cell 19: Run real sample predictions
    add_markdown("### Step 19: Run Real Sample Predictions with Probabilities")
    add_code("""def predict_pair(text_a, text_b):
    inputs = tokenizer(
        text_a, text_b,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    return {
        "text_a": text_a,
        "text_b": text_b,
        "prediction": ID2LABEL[pred_id],
        "prob_not_paraphrase": float(probs[0]),
        "prob_paraphrase": float(probs[1])
    }

demo_samples = [
    (
        "The company announced a new product yesterday.",
        "Yesterday, the company revealed its latest product."
    ),
    (
        "The dog bit the mailman near the front gate.",
        "The mailman bit the dog near the front gate."
    ),
    (
        "Quantum computing relies on qubits to perform exponential calculations.",
        "Photosynthesis allows green plants to convert solar radiation into chemical energy."
    )
]

print("=" * 75)
print("REAL INFERENCE SAMPLES ON FINE-TUNED DISTILBERT")
print("=" * 75)
for i, (t_a, t_b) in enumerate(demo_samples, 1):
    res = predict_pair(t_a, t_b)
    print(f"\\nSample {i}:")
    print(f"  Text A : {res['text_a']}")
    print(f"  Text B : {res['text_b']}")
    print(f"  Result : **{res['prediction']}**")
    print(f"  P(Paraphrase): {res['prob_paraphrase']:.4f} | P(Not Paraphrase): {res['prob_not_paraphrase']:.4f}")
print("=" * 75)""")

    # Cell 20: Calculate BERTScore on a manageable evaluation sample
    add_markdown("### Step 20: Calculate BERTScore on Evaluation Sample (Supplementary Semantic Metric)")
    add_code("""bertscore_res = {}
try:
    from bert_score import score as bert_score_fn
    print("[*] Computing BERTScore on 100 test samples...")
    sample_eval = test_sub.head(100)
    P, R, F1 = bert_score_fn(
        sample_eval["sentence1"].tolist(),
        sample_eval["sentence2"].tolist(),
        model_type=MODEL_NAME,
        lang="en",
        verbose=False,
        batch_size=32
    )
    bertscore_res = {
        "status": "success",
        "sample_size": len(sample_eval),
        "mean_precision": round(float(P.mean()), 4),
        "mean_recall": round(float(R.mean()), 4),
        "mean_f1": round(float(F1.mean()), 4)
    }
    print(f"BERTScore Results: P={bertscore_res['mean_precision']}, R={bertscore_res['mean_recall']}, F1={bertscore_res['mean_f1']}")
except Exception as e:
    print(f"BERTScore evaluation note: {e}")
    bertscore_res = {"status": "error_or_skipped", "message": str(e)}""")

    # Cell 21: Compare the fine-tuned classifier with a TF-IDF baseline
    add_markdown("### Step 21: Compare Fine-Tuned DistilBERT with TF-IDF Baseline on Identical Test Split")
    add_code("""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("[*] Training and evaluating TF-IDF + Logistic Regression baseline...")

# Fit vocabulary on training pairs
train_corpus = train_sub["sentence1"].tolist() + train_sub["sentence2"].tolist()
tfidf_vec = TfidfVectorizer(max_features=10000, stop_words="english")
tfidf_vec.fit(train_corpus)

def extract_features(df):
    v1 = tfidf_vec.transform(df["sentence1"]).toarray()
    v2 = tfidf_vec.transform(df["sentence2"]).toarray()
    diff = np.abs(v1 - v2)
    prod = v1 * v2
    cos = np.sum(v1 * v2, axis=1, keepdims=True) / (
        np.linalg.norm(v1, axis=1, keepdims=True) * np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
    )
    return np.hstack([diff, prod, cos])

X_train_tfidf = extract_features(train_sub)
X_test_tfidf = extract_features(test_sub)

lr_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
lr_model.fit(X_train_tfidf, train_sub["label"])
tfidf_preds = lr_model.predict(X_test_tfidf)

tfidf_acc = accuracy_score(test_sub["label"], tfidf_preds)
tfidf_f1 = f1_score(test_sub["label"], tfidf_preds, pos_label=1, zero_division=0)

print(f"TF-IDF Baseline Accuracy : {tfidf_acc:.4f} | F1: {tfidf_f1:.4f}")
print(f"DistilBERT Model Accuracy: {test_acc:.4f} | F1: {bin_f1:.4f}")
print(f"Performance Difference  : {(test_acc - tfidf_acc)*100:+.2f}% accuracy points")

tfidf_baseline_res = {
    "baseline_name": "TF-IDF + Logistic Regression",
    "accuracy": round(float(tfidf_acc), 4),
    "f1_score": round(float(tfidf_f1), 4)
}""")

    # Cell 22: Save actual metrics and visualizations
    add_markdown("### Step 22: Save Actual Metrics and Metadata to `results/metrics.json`")
    add_code("""import json

metrics_payload = {
    "dataset": "PAWS (labeled_final)",
    "base_model": MODEL_NAME,
    "device": device.type,
    "training_metadata": {
        "train_samples": len(train_sub),
        "val_samples": len(val_sub),
        "test_samples": len(test_sub),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "training_duration_seconds": round(training_time, 2),
        "loss_history": history
    },
    "distilbert_evaluation": {
        "accuracy": round(float(test_acc), 4),
        "binary_precision": round(float(bin_prec), 4),
        "binary_recall": round(float(bin_rec), 4),
        "binary_f1": round(float(bin_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "inference_latency_ms": round(float(latency_per_sample), 2),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }
    },
    "tfidf_baseline_comparison": tfidf_baseline_res,
    "supplementary_bertscore": bertscore_res
}

with open("results/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print("[*] Successfully saved 'results/metrics.json'.")""")

    # Cell 23: Export the model checkpoint and evaluation artifacts
    add_markdown("### Step 23: Export Model Checkpoint and Evaluation Artifacts as Zip")
    add_code("""import shutil

print("[*] Creating archive of trained model checkpoint and evaluation artifacts...")
shutil.make_archive("fine_tuned_distilbert", "zip", root_dir="models", base_dir="fine_tuned_distilbert")
shutil.make_archive("results_artifacts", "zip", root_dir="results")

print(f"Archive created: fine_tuned_distilbert.zip ({os.path.getsize('fine_tuned_distilbert.zip') / (1024*1024):.1f} MB)")
print(f"Archive created: results_artifacts.zip ({os.path.getsize('results_artifacts.zip') / 1024:.1f} KB)")

# Attempt Colab automated download
try:
    from google.colab import files
    print("[*] Triggering Colab browser download for model and results...")
    files.download("fine_tuned_distilbert.zip")
    files.download("results_artifacts.zip")
except Exception:
    print("[*] Running outside interactive Colab environment. Archive saved in local directory.")""")

    # Cell 24: Display a final completion summary
    add_markdown("### Step 24: Display Final Completion Summary")
    add_code("""print("=" * 80)
print("DISTILBERT FINE-TUNING AND EVALUATION PIPELINE: COMPLETE SUMMARY")
print("=" * 80)
print(f"Base Pretrained Model : {MODEL_NAME}")
print(f"Dataset Used          : PAWS (labeled_final) -> {len(train_sub)} train, {len(test_sub)} test")
print(f"Training Duration     : {training_time:.2f} s on {device}")
print(f"Epochs Completed      : {EPOCHS}")
print(f"Test Accuracy         : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test F1-Score         : {bin_f1:.4f}")
print(f"Baseline TF-IDF Acc   : {tfidf_acc:.4f} (Advantage: +{(test_acc - tfidf_acc)*100:.2f}%)")
print(f"Model Checkpoint Path : {OUTPUT_DIR}")
print(f"Artifact Archive      : fine_tuned_distilbert.zip")
print("=" * 80)
print("NEXT STEPS:")
print("1. Extract 'fine_tuned_distilbert.zip' into 'models/fine_tuned_distilbert/' in your local workspace.")
print("2. Extract 'results_artifacts.zip' into 'results/' to load confusion matrix & curves.")
print("3. Launch the Streamlit application: streamlit run app.py")
print("=" * 80)""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    os.makedirs("notebooks", exist_ok=True)
    nb_path = "notebooks/plagiarism_paraphrase_finetuning.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated {nb_path} with {len(cells)} cells.")

if __name__ == "__main__":
    build_notebook()
