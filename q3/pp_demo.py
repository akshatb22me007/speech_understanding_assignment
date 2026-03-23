
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from privacymodule import PrivacyAnonymizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/cv-valid-train.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/cv-valid-train"))
    parser.add_argument("--source", type=str, default="clip_001.wav")
    parser.add_argument("--target", type=str, default="clip_002.wav")
    parser.add_argument("--out-dir", type=Path, default=Path("examples"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.audio_dir / args.source
    target_path = args.audio_dir / args.target

    source_wav, sr = librosa.load(source_path, sr=16000, mono=True)
    target_wav, _ = librosa.load(target_path, sr=16000, mono=True)

    module = PrivacyAnonymizer()
    anonymized, sr, params = module.anonymize_waveform(
        source_wav,
        sr=sr,
        source_ref=source_wav,
        target_ref=target_wav,
    )

    sf.write(args.out_dir / "original.wav", source_wav, sr)
    sf.write(args.out_dir / "anonymized.wav", anonymized, sr)
    sf.write(args.out_dir / "target_reference.wav", target_wav, sr)

    print("Saved:")
    print(f"  original.wav -> {args.out_dir / 'original.wav'}")
    print(f"  anonymized.wav -> {args.out_dir / 'anonymized.wav'}")
    print(f"  target_reference.wav -> {args.out_dir / 'target_reference.wav'}")
    print(f"Params: semitones={params.semitones:.2f}, rate={params.rate:.3f}, tilt={params.tilt:.3f}")


if __name__ == "__main__":
    main()
