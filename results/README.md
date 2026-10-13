# Evaluation Results and Artifacts

This directory stores authentic, empirically generated evaluation outputs produced after running model training and testing on the PAWS held-out test split.

## Expected Artifacts

When evaluation is executed via `notebooks/plagiarism_paraphrase_finetuning.ipynb` or `train.py`, the following artifacts are generated:

1. `metrics.json`:
   - Final test loss, test accuracy, binary precision, binary recall, and binary F1-score.
   - Macro-averaged precision, recall, and F1-score.
   - Inference latency per sample.
   - Training metadata (dataset version, subset sizes, epochs, batch size, learning rate, hardware device used, execution duration).
   - TF-IDF baseline comparative metrics on the identical test split.
   - BERTScore semantic evaluation metrics (precision, recall, F1) on the evaluation sample if computed.

2. `classification_report.txt`:
   - Full scikit-learn classification report broken down by class:
     - Class 0: Not Paraphrase
     - Class 1: Paraphrase
   - Macro average and weighted average metrics.

3. `confusion_matrix.png`:
   - Heatmap plot illustrating True Negatives, False Positives, False Negatives, and True Positives on the held-out test set.

4. `training_curves.png`:
   - Epoch-by-epoch or step-by-step training loss vs. validation loss curve.

## Academic Integrity Notice
These files are generated strictly from live execution on held-out test data. If training has not yet been executed in your local environment, run the fine-tuning notebook in Google Colab (or execute `train.py`) to generate these real evaluation artifacts.
