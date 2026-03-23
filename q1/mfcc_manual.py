import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

from audio_utils import frame_signal, load_audio, pre_emphasis, window_fn


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_fft: int,
    sr: int,
    n_mels: int = 26,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0

    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1

        for k in range(left, center):
            fb[i - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[i - 1, k] = (right - k) / max(right - center, 1)
    return fb


def manual_mfcc(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    n_mels: int = 26,
    n_ceps: int = 13,
    preemph: float = 0.97,
    window: str = "hamming",
) -> dict:
    x = pre_emphasis(x, preemph)
    frames, frame_len, hop_len = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)

    win = window_fn(window, frame_len)
    frames_w = frames * win[None, :]

    spec = np.fft.rfft(frames_w, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2) / n_fft

    fb = mel_filterbank(n_fft=n_fft, sr=sr, n_mels=n_mels)
    mel_energy = np.dot(power, fb.T)
    mel_energy = np.maximum(mel_energy, 1e-12)
    log_mel = np.log(mel_energy)

    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_ceps]
    cepstrum = np.fft.irfft(np.log(np.maximum(np.abs(spec), 1e-12)), axis=1)

    return {
        "mfcc": mfcc,
        "log_mel": log_mel,
        "power": power,
        "cepstrum": cepstrum,
        "frame_len": frame_len,
        "hop_len": hop_len,
        "window": win,
        "filterbank": fb,
    }


def plot_outputs(out_dir: Path, res: Dict[str, np.ndarray], sr: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.imshow(res["log_mel"].T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="Log Mel Energy")
    plt.title("Manual Log-Mel Spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel Bin")
    plt.tight_layout()
    plt.savefig(out_dir / "manual_logmel.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(res["mfcc"].T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="MFCC")
    plt.title("Manual MFCC (DCT of Log-Mel)")
    plt.xlabel("Frame")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_dir / "manual_mfcc.png", dpi=150)
    plt.close()

    # Visualize low/high quefrency energy trends.
    cep = np.abs(res["cepstrum"])
    q_low = int(0.002 * sr)
    q_high_lo = int(0.003 * sr)
    q_high_hi = int(0.015 * sr)
    low = cep[:, 1:q_low].mean(axis=1)
    high = cep[:, q_high_lo:q_high_hi].mean(axis=1)

    t = np.arange(len(low)) * (res["hop_len"] / sr)
    plt.figure(figsize=(10, 4))
    plt.plot(t, low, label="Low-Quefrency Envelope")
    plt.plot(t, high, label="High-Quefrency Periodicity")
    plt.title("Cepstral Regions")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean |Cepstrum|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cepstral_regions.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual MFCC/Cepstrum extraction")
    ap.add_argument("--audio", required=True, help="Path to audio file")
    ap.add_argument("--out_dir", default="output/mfcc", help="Output directory")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    ap.add_argument("--frame_ms", type=float, default=25.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--n_mels", type=int, default=26)
    ap.add_argument("--n_ceps", type=int, default=13)
    ap.add_argument("--window", type=str, default="hamming", choices=["rectangular", "hamming", "hanning"])
    args = ap.parse_args()

    x, sr = load_audio(args.audio, target_sr=args.sr)
    res = manual_mfcc(
        x,
        sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        n_ceps=args.n_ceps,
        window=args.window,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "mfcc.npy", res["mfcc"])
    np.save(out_dir / "log_mel.npy", res["log_mel"])
    np.save(out_dir / "cepstrum.npy", res["cepstrum"])
    np.save(out_dir / "power.npy", res["power"])

    plot_outputs(out_dir, res, sr)
    print(f"Saved MFCC/Cepstrum outputs to: {out_dir}")
    print(f"MFCC shape: {res['mfcc'].shape}, Log-Mel shape: {res['log_mel'].shape}")


if __name__ == "__main__":
    main()