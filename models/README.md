# Models Directory

This directory stores the fine-tuned DistilBERT model checkpoints, tokenizer configurations, and label mappings.

## Checkpoint Structure

After running either the Google Colab training notebook (`notebooks/plagiarism_paraphrase_finetuning.ipynb`) or the local training script (`train.py`), export or place the checkpoint files inside the subfolder:

```
models/fine_tuned_distilbert/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── training_args.bin (optional)
```

## How to Import a Trained Checkpoint from Google Colab

1. Run all cells in `notebooks/plagiarism_paraphrase_finetuning.ipynb` on Google Colab (with T4 GPU).
2. The notebook automatically bundles the checkpoint into `fine_tuned_distilbert.zip` and triggers a download (or saves to Google Drive).
3. Extract the contents into this directory:
   ```bash
   unzip fine_tuned_distilbert.zip -d models/fine_tuned_distilbert/
   ```
4. Verify that `models/fine_tuned_distilbert/config.json` exists.
5. Launch or refresh the Streamlit interface (`streamlit run app.py`). The system will automatically detect and load the fine-tuned checkpoint.

## Note on Model Integrity
In strict adherence to academic honesty and engineering integrity, this repository never commits heavy binary weights to Git and never silently replaces a missing fine-tuned checkpoint with an untrained base model.
