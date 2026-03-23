import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 in [-1, 1], optionally resampled."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".wav":
        sr, data = wavfile.read(str(p))
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
    else:
        # soundfile is used only for decoding non-WAV formats.
        import soundfile as sf

        data, sr = sf.read(str(p), always_2d=False)
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)
        sr = target_sr

    peak = np.max(np.abs(data)) + 1e-12
    if peak > 1.0:
        data = data / peak

    return data, sr


def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def frame_signal(
    x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> Tuple[np.ndarray, int, int]:
    frame_len = int(round(frame_ms * sr / 1000.0))
    hop_len = int(round(hop_ms * sr / 1000.0))
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_ms/hop_ms produce invalid frame sizes")

    if len(x) < frame_len:
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad), mode="constant")

    n_frames = 1 + int(np.ceil((len(x) - frame_len) / hop_len))
    total_len = (n_frames - 1) * hop_len + frame_len
    pad_len = total_len - len(x)
    if pad_len > 0:
        x = np.pad(x, (0, pad_len), mode="constant")

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(0, n_frames * hop_len, hop_len), (frame_len, 1)).T
    )
    frames = x[idx]
    return frames, frame_len, hop_len


def window_fn(name: str, frame_len: int) -> np.ndarray:
    n = np.arange(frame_len)
    key = name.lower()
    if key == "rectangular":
        return np.ones(frame_len, dtype=np.float32)
    if key == "hamming":
        return (0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_len - 1))).astype(np.float32)
    if key in ("hanning", "hann"):
        return (0.5 - 0.5 * np.cos(2 * np.pi * n / (frame_len - 1))).astype(np.float32)
    raise ValueError(f"Unknown window: {name}")


def frame_times(n_frames: int, hop_len: int, sr: int) -> np.ndarray:
    return np.arange(n_frames) * hop_len / float(sr)


def read_manifest(manifest_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in manifest: {manifest_path}")
    return rows