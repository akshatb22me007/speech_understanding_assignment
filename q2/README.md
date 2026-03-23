# Q2 Speech Experiment Reproduction Guide

## Environment
- Python >= 3.10
- Install deps (example): `pip install torch torchaudio soundfile pyyaml scikit-learn matplotlib`
- Run from repo root:
  - Windows PowerShell: `cd e:\speech_assignment\q2`
  - Linux/macOS: `cd /path/to/speech_assignment/q2`

## Data
- Uses LibriSpeech `dev-clean` laid out under `data/LibriSpeech/dev-clean` (already referenced in `configs/config.yaml`).
- The loader expects speaker-wise folders and recursively reads `.flac`/`.wav` files.
```bash
mkdir data
cd data

# small dataset
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xvf dev-clean.tar.gz
```

## Training
- Config: `configs/config.yaml` (current defaults in repo):
  - `train_split: 0.8`
  - `seed: 42`
  - `batch_size: 32`
  - `epochs: 50`
  - `lr: 1e-4`
  - `device: cuda`
- Command: `python train.py`
- Output: results/model.pth (feature_extractor + disentangler weights) and results/split_indices.pt (train/test indices for reproducibility).
- Also saves training curve: `results/training_loss.png`.

## Evaluation
- Command: `python eval.py`
- Uses saved `split_indices.pt` to evaluate only on the held-out test subset; falls back to deterministic split if missing.
- Outputs:
  - Prints EER, ROC-AUC, PR-AUC, Accuracy@EER threshold, and F1@EER threshold.
  - Saves metrics to `results/eval_results.txt`.
  - Saves plots:
    - `results/roc_curve.png`
    - `results/pr_curve.png`
    - `results/score_distribution.png`
    - `results/confusion_matrix.png`

## Improved Experiment
- Command: `python improved_exp.py`
- Main change vs base training: adds contrastive term on speaker code
  - `loss = reconstruction_loss + classification_loss + 0.5 * contrastive_loss`
- Outputs:
  - Checkpoint: `results/improved_model.pth`
  - Metrics: `results/improved_results.txt`
  - Plots:
    - `results/improved_training_loss.png`
    - `results/improved_roc_curve.png`
    - `results/improved_pr_curve.png`
    - `results/improved_score_distribution.png`
    - `results/improved_confusion_matrix.png`

## Reported Results (Current Artifacts)

| Metric | Base (`eval_results.txt`) | Improved (`improved_results.txt`) |
|---|---:|---:|
| EER | 0.2833 | 0.1791 |
| ROC AUC | 0.7966 | 0.9093 |
| PR AUC | 0.1119 | 0.2625 |
| Accuracy@EER threshold | 0.7167 | 0.8208 |
| Precision@EER threshold | 0.0624 | 0.1075 |
| Recall@EER threshold | 0.7167 | 0.8210 |
| F1@EER threshold | 0.1148 | 0.1901 |

Confusion matrix counts at EER threshold:
- Base: TN=102011, FP=40317, FN=1060, TP=2682
- Improved: TN=116829, FP=25499, FN=670, TP=3072

## Reproducibility Notes
- `train.py` and `improved_exp.py` both save split indices to `results/split_indices.pt`.
- `eval.py` reuses the same test split when this file exists.
- Deterministic split is controlled by `seed` in `configs/config.yaml`.
- If running on CPU, set `device: cpu` in `configs/config.yaml` before training/eval.

## Checkpoint provenance
- `results/model.pth` corresponds to base training with `configs/config.yaml` defaults in this repo.
- `results/improved_model.pth` corresponds to contrastive-loss training (`improved_exp.py`) using the same data/split setup.
