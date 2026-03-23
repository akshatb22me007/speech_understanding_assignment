
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
from scipy.linalg import sqrtm


def audio_embedding(path: Path, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)])
    return feats.astype(np.float64)


def stats(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = embeddings.mean(axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def fad_from_paths(original: Iterable[Path], generated: Iterable[Path]) -> float:
    orig = np.stack([audio_embedding(p) for p in original], axis=0)
    gen = np.stack([audio_embedding(p) for p in generated], axis=0)

    mu1, sigma1 = stats(orig)
    mu2, sigma2 = stats(gen)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fid))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", nargs="+", type=Path, required=True)
    parser.add_argument("--generated", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    print("FAD:", fad_from_paths(args.original, args.generated))


if __name__ == "__main__":
    main()
