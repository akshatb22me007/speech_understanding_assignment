import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_utils import frame_signal, load_audio, pre_emphasis, window_fn

def get_main_lobe_bounds(mag2, peak, threshold_db=-20):
    peak_val = mag2[peak]
    threshold = peak_val * (10 ** (threshold_db / 10.0))

    left = peak
    while left > 0 and mag2[left] > threshold:
        left -= 1

    right = peak
    while right < len(mag2) - 1 and mag2[right] > threshold:
        right += 1

    return left, right

def leakage_and_snr(frame: np.ndarray, window: np.ndarray, n_fft: int) -> Tuple[float, float, np.ndarray]:
    xw = frame * window
    mag2 = np.abs(np.fft.rfft(xw, n=n_fft)) ** 2
    total_power = float(np.sum(mag2))

    # If the frame is effectively silent, avoid divide-by-zero and signal an unusable measurement.
    if total_power <= 1e-12:
        return 1.0, float("-inf"), mag2

    peak = int(np.argmax(mag2))
    left, right = get_main_lobe_bounds(mag2, peak)
    main_lobe_power = float(np.sum(mag2[left : right + 1]))
    leakage_power = max(total_power - main_lobe_power, 1e-12)

    leakage_ratio = leakage_power / total_power
    snr_db = 10.0 * np.log10(main_lobe_power / leakage_power)
    return leakage_ratio, snr_db, mag2


def analyze_windows(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 1024,
    frame_index: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    x = pre_emphasis(x)
    frames, frame_len, _ = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)

    # Pick a representative frame: default to the maximum-energy frame to avoid silent padding.
    energies = np.sum(frames**2, axis=1)
    if frame_index < 0:
        idx = int(np.argmax(energies))
    else:
        idx = min(max(frame_index, 0), len(frames) - 1)
        # If the chosen frame is nearly silent, fall back to the max-energy frame.
        if energies[idx] <= 1e-8:
            idx = int(np.argmax(energies))

    frame = frames[idx]

    results = []
    spectra: Dict[str, np.ndarray] = {}
    for name in ["rectangular", "hamming", "hanning"]:
        w = window_fn(name, frame_len)
        leakage, snr_db, mag2 = leakage_and_snr(frame, w, n_fft=n_fft)
        spectra[name] = mag2
        results.append(
            {
                "window": name,
                "spectral_leakage_ratio": leakage,
                "spectral_leakage_percent": 100.0 * leakage,
                "snr_db": snr_db,
            }
        )

    df = pd.DataFrame(results).sort_values(by="snr_db", ascending=False).reset_index(drop=True)
    return df, spectra


def plot_results(df: pd.DataFrame, spectra: Dict[str, np.ndarray], sr: int, n_fft: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "leakage_snr_table.csv", index=False)
    print(df.to_string(index=False))

    plt.figure(figsize=(9, 4))
    x = np.arange(len(df))
    plt.bar(x - 0.18, df["spectral_leakage_percent"], width=0.36, label="Leakage %")
    plt.bar(x + 0.18, df["snr_db"], width=0.36, label="SNR (dB)")
    plt.xticks(x, df["window"])
    plt.ylabel("Value")
    plt.title("Window Comparison: Spectral Leakage and SNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "leakage_snr_comparison.png", dpi=150)
    plt.close()

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    plt.figure(figsize=(10, 5))
    for name, mag2 in spectra.items():
        plt.plot(freqs, 10 * np.log10(mag2 + 1e-12), label=name)
    plt.xlim(0, min(4000, sr // 2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title("Spectrum Under Different Windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "window_spectra.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Spectral leakage and SNR window analysis")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="output/leakage_snr")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=25.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument(
        "--frame_index",
        type=int,
        default=-1,
        help="Frame to analyze (-1 picks the highest-energy frame)",
    )
    args = ap.parse_args()

    x, sr = load_audio(args.audio, target_sr=args.sr)
    df, spectra = analyze_windows(
        x,
        sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        frame_index=args.frame_index,
    )
    plot_results(df, spectra, sr=sr, n_fft=args.n_fft, out_dir=Path(args.out_dir))
    print(f"Saved leakage/SNR analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()