# Q2 Speech Experiment Reproduction Guide

## Environment
- Python >= 3.10
- Install deps (example): `pip install torch torchaudio soundfile pyyaml scikit-learn matplotlib`
- Run from repo root: `cd /data2/autoNav/akshat/speech/q2_final`

## Data
- Uses LibriSpeech dev-clean laid out under data/LibriSpeech/dev-clean (already referenced in configs/config.yaml).

## Training
- Config: configs/config.yaml (key fields: train_split=0.8, seed=42, batch_size=8, epochs=20, lr=1e-4, device=cuda).
- Command: `python train.py`
- Output: results/model.pth (feature_extractor + disentangler weights) and results/split_indices.pt (train/test indices for reproducibility).

## Evaluation
- Command: `python eval.py`
- Uses saved split_indices.pt to evaluate only on the held-out test subset; falls back to deterministic split if missing.
- Outputs:
  - Prints EER to console.
  - Saves ROC curve plot to results/eval_roc.png.

## Checkpoint provenance
- results/model.pth corresponds to training with configs/config.yaml defaults (train_split=0.8, seed=42, epochs=20) on data/LibriSpeech/dev-clean.
