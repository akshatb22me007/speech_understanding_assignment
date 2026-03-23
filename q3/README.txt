
q3 Offline Package
==================

This bundle is a self-contained offline demo for the Common Voice-style
ethics/fairness assignment.

Files:
- audit.py
- privacymodule.py
- pp_demo.py
- train_fair.py
- evaluation_scripts/dnsmos.py
- evaluation_scripts/fad.py
- examples/original.wav
- examples/anonymized.wav
- audit_plots.pdf
- q3_report.tex

Notes:
- The code defaults to the local demo dataset in q3/data/.
- If you provide the real Kaggle Common Voice CSV/audio folders, the same
  scripts can be pointed at those paths with --csv and --audio-dir.
- The DNSMOS script uses a proxy when the ONNX runtime/model is unavailable.
- The ASR script uses wav2vec2 only if local cached weights exist; otherwise it
  falls back to an offline TinyASR CTC model.
