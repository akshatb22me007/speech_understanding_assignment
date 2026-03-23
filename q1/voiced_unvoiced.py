import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_utils import frame_signal, frame_times, load_audio, pre_emphasis, window_fn


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, ker, mode="valid")


def segment_voiced_unvoiced(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    window: str = "hamming",
    smooth_k: int = 5,
) -> dict:
    x = pre_emphasis(x)
    frames, frame_len, hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    win = window_fn(window, frame_len)
    frames_w = frames * win[None, :]

    spec = np.fft.rfft(frames_w, n=n_fft, axis=1)
    log_mag = np.log(np.maximum(np.abs(spec), 1e-12))
    cep = np.fft.irfft(log_mag, axis=1)
    cep_abs = np.abs(cep)

    q_low = int(0.002 * sr)
    q_high_lo = int(0.003 * sr)
    q_high_hi = int(0.015 * sr)

    low_env = cep_abs[:, 1:max(q_low, 2)].mean(axis=1)
    high_pitch = cep_abs[:, max(q_high_lo, 2):max(q_high_hi, q_high_lo + 1)].mean(axis=1)
    score = high_pitch / (low_env + 1e-9)
    score_sm = moving_average(score, smooth_k)

    thresh = float(np.percentile(score_sm, 55.0))
    voiced = (score_sm >= thresh).astype(np.int32)

    # Remove very short flips.
    min_frames = max(2, int(round(0.03 / (hop_len / sr))))
    voiced_clean = voiced.copy()
    s = 0
    while s < len(voiced_clean):
        e = s + 1
        while e < len(voiced_clean) and voiced_clean[e] == voiced_clean[s]:
            e += 1
        if (e - s) < min_frames:
            left = voiced_clean[s - 1] if s > 0 else voiced_clean[e] if e < len(voiced_clean) else voiced_clean[s]
            voiced_clean[s:e] = left
        s = e

    t = frame_times(len(frames), hop_len, sr)
    return {
        "times": t,
        "low_env": low_env,
        "high_pitch": high_pitch,
        "score": score_sm,
        "threshold": thresh,
        "labels": voiced_clean,
        "hop_len": hop_len,
    }


def labels_to_segments(times: np.ndarray, labels: np.ndarray, hop_sec: float) -> List[Dict[str, Any]]:
    segs = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segs.append(
                {
                    "start_sec": float(times[start]),
                    "end_sec": float(times[i] + hop_sec),
                    "label": "voiced" if labels[i - 1] == 1 else "unvoiced",
                }
            )
            start = i
    segs.append(
        {
            "start_sec": float(times[start]),
            "end_sec": float(times[-1] + hop_sec),
            "label": "voiced" if labels[-1] == 1 else "unvoiced",
        }
    )
    return segs


def plot_results(out_dir: Path, x: np.ndarray, sr: int, res: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_audio = np.arange(len(x)) / float(sr)

    plt.figure(figsize=(11, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_audio, x, linewidth=0.8)
    ax1.set_title("Waveform with Voiced/Unvoiced Segments")
    ax1.set_ylabel("Amplitude")

    hop_sec = res["hop_len"] / sr
    segs = labels_to_segments(res["times"], res["labels"], hop_sec)
    for seg in segs:
        color = "#89CFF0" if seg["label"] == "unvoiced" else "#FFB347"
        ax1.axvspan(seg["start_sec"], seg["end_sec"], color=color, alpha=0.25)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(res["times"], res["score"], label="High/Low Quefrency Ratio", color="black")
    ax2.axhline(res["threshold"], color="red", linestyle="--", label="Threshold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cepstral Score")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "voiced_unvoiced_segmentation.png", dpi=150)
    plt.close()

    pd.DataFrame(segs).to_csv(out_dir / "voiced_unvoiced_segments.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Voiced/unvoiced segmentation from cepstral regions")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="output/voiced_unvoiced")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=25.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=512)
    args = ap.parse_args()

    x, sr = load_audio(args.audio, target_sr=args.sr)
    res = segment_voiced_unvoiced(
        x,
        sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
    )
    out_dir = Path(args.out_dir)
    plot_results(out_dir, x, sr, res)
    print(f"Saved voiced/unvoiced outputs to: {out_dir}")


if __name__ == "__main__":
    main()