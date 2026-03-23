import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from audio_utils import load_audio
from voiced_unvoiced import labels_to_segments, segment_voiced_unvoiced


def ctc_token_alignment(
    audio: np.ndarray,
    sr: int,
    model_name: str,
    device: str,
) -> List[Dict[str, Any]]:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits[0]

    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    vocab = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    blank_id = processor.tokenizer.pad_token_id

    n_frames = len(pred_ids)
    dur = len(audio) / float(sr)
    frame_dt = dur / max(n_frames, 1)

    segments = []
    s = 0
    while s < n_frames:
        e = s + 1
        while e < n_frames and pred_ids[e] == pred_ids[s]:
            e += 1
        tok_id = int(pred_ids[s])
        tok = id_to_token.get(tok_id, "")
        if tok_id != blank_id and tok not in {"<s>", "</s>", "<pad>", "|"} and tok.strip() != "":
            tok = tok.replace(" ", "")
            segments.append(
                {
                    "start_sec": s * frame_dt,
                    "end_sec": e * frame_dt,
                    "phone": tok,
                }
            )
        s = e

    merged = []
    for seg in segments:
        if merged and merged[-1]["phone"] == seg["phone"] and seg["start_sec"] <= merged[-1]["end_sec"] + 1e-3:
            merged[-1]["end_sec"] = seg["end_sec"]
        else:
            merged.append(seg)
    return merged


def boundary_rmse(manual_bounds: np.ndarray, model_bounds: np.ndarray) -> float:
    if len(manual_bounds) == 0 or len(model_bounds) == 0:
        return float("nan")
    diffs = []
    for b in manual_bounds:
        nearest = model_bounds[np.argmin(np.abs(model_bounds - b))]
        diffs.append((b - nearest) ** 2)
    return float(np.sqrt(np.mean(diffs)))


def map_segments_to_phones(
    manual_segments: List[Dict[str, Any]],
    phone_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mapped = []
    for ms in manual_segments:
        overlaps = []
        for ps in phone_segments:
            st = max(ms["start_sec"], ps["start_sec"])
            en = min(ms["end_sec"], ps["end_sec"])
            ov = max(0.0, en - st)
            if ov > 0:
                overlaps.append((ov, ps["phone"]))
        if overlaps:
            phone = sorted(overlaps, key=lambda x: x[0], reverse=True)[0][1]
        else:
            phone = "<none>"
        mapped.append(
            {
                "start_sec": ms["start_sec"],
                "end_sec": ms["end_sec"],
                "label": ms["label"],
                "mapped_phone": phone,
            }
        )
    return mapped


def plot_alignment(
    manual_segments: List[Dict[str, Any]],
    phones: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 3))
    for seg in manual_segments:
        color = "#F4A261" if seg["label"] == "voiced" else "#8ECae6"
        plt.axvspan(seg["start_sec"], seg["end_sec"], ymin=0.55, ymax=0.95, color=color, alpha=0.5)

    for ph in phones:
        plt.axvspan(ph["start_sec"], ph["end_sec"], ymin=0.05, ymax=0.45, color="#90BE6D", alpha=0.35)
        plt.text((ph["start_sec"] + ph["end_sec"]) / 2, 0.02, ph["phone"], ha="center", va="bottom", fontsize=8)

    plt.yticks([0.25, 0.75], ["Model phones", "Manual V/UV"])
    plt.ylim(0, 1)
    plt.xlabel("Time (s)")
    plt.title("Manual Segments vs Hugging Face Token Alignment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Map manual voiced/unvoiced boundaries to model phones")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="output/phonetic_mapping")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--model_name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, sr = load_audio(args.audio, target_sr=args.sr)

    vu = segment_voiced_unvoiced(x, sr)
    hop_sec = vu["hop_len"] / sr
    manual_segments = labels_to_segments(vu["times"], vu["labels"], hop_sec)
    phone_segments = ctc_token_alignment(x, sr, model_name=args.model_name, device=args.device)

    manual_bounds = np.array([s["start_sec"] for s in manual_segments[1:]], dtype=np.float32)
    model_bounds = np.array([s["start_sec"] for s in phone_segments[1:]], dtype=np.float32)
    rmse_sec = boundary_rmse(manual_bounds, model_bounds)

    mapped = map_segments_to_phones(manual_segments, phone_segments)
    pd.DataFrame(mapped).to_csv(out_dir / "manual_segments_with_phones.csv", index=False)
    pd.DataFrame(phone_segments).to_csv(out_dir / "model_phone_segments.csv", index=False)

    metrics = pd.DataFrame([{"rmse_sec": rmse_sec, "rmse_ms": rmse_sec * 1000.0}])
    metrics.to_csv(out_dir / "rmse_metrics.csv", index=False)

    plot_alignment(manual_segments, phone_segments, out_dir / "alignment_plot.png")

    print(f"Saved phonetic mapping outputs to: {out_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()