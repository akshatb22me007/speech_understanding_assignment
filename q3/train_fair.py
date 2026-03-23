
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False


def simple_wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(1, n)


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def extract_mel_mean(wav: np.ndarray, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=n_mels,
        fmin=50,
        fmax=min(7600, sr // 2),
        power=2.0,
    )
    mel = np.log1p(mel).astype(np.float32)
    return mel.mean(axis=1)  # (n_mels,)


@dataclass
class Sample:
    path: Path
    text: str
    gender: str


class TinyPhraseASR(nn.Module):
    def __init__(self, n_mels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def group_means(values: List[torch.Tensor], groups: List[str]) -> torch.Tensor:
    out = []
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if idx:
            out.append(torch.stack([values[i] for i in idx]).mean())
    return torch.stack(out) if len(out) > 1 else torch.tensor([0.0], device=values[0].device)


def try_wav2vec2(csv_path: Path, audio_dir: Path) -> bool:
    if not _HAVE_TRANSFORMERS:
        return False
    try:
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h",
            local_files_only=True,
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h",
            local_files_only=True,
        )
    except Exception:
        return False

    df = pd.read_csv(csv_path)
    df = df[df["up_votes"] > df["down_votes"]].head(1)
    sample = df.iloc[0]
    wav = load_audio(audio_dir / sample["filename"])
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    text = processor.batch_decode(logits.argmax(dim=-1))[0]
    print("Wav2Vec2 demo transcription:", text)
    return True


def train_fallback(csv_path: Path, audio_dir: Path, out_path: Path, steps: int = 120, lambda_fair: float = 0.25) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df["up_votes"] > df["down_votes"]].copy()

    samples = [
        Sample(
            path=audio_dir / row["filename"],
            text=str(row["text"]),
            gender=str(row["gender"]) if pd.notna(row["gender"]) else "unknown",
        )
        for _, row in df.iterrows()
    ]

    phrases = sorted(set(s.text for s in samples))
    phrase_to_id = {p: i for i, p in enumerate(phrases)}
    id_to_phrase = {i: p for p, i in phrase_to_id.items()}

    X = []
    y = []
    g = []
    for s in samples:
        wav = load_audio(s.path)
        X.append(extract_mel_mean(wav))
        y.append(phrase_to_id[s.text])
        g.append(s.gender)

    X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    model = TinyPhraseASR(n_mels=X.shape[1], n_classes=len(phrases)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)

    for step in range(1, steps + 1):
        logits = model(X)
        losses = F.cross_entropy(logits, y, reduction="none")
        group_loss_means = group_means([losses[i] for i in range(len(samples))], g)
        fairness = torch.var(group_loss_means) if group_loss_means.numel() > 1 else torch.tensor(0.0, device=device)
        loss = losses.mean() + lambda_fair * fairness

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 30 == 0 or step == 1 or step == steps:
            print(f"step={step:03d} loss={loss.item():.4f} ctc_proxy={losses.mean().item():.4f} fair={fairness.item():.4f}")

    with torch.no_grad():
        pred_ids = model(X).argmax(dim=-1).cpu().tolist()

    results = []
    for s, pid in zip(samples, pred_ids):
        hyp = id_to_phrase[pid]
        results.append((s.gender, s.text, hyp, simple_wer(s.text, hyp)))

    group_wers = {}
    for gender in sorted(set(g for g in g if isinstance(g, str) and g)):
        vals = [wer for gg, _, _, wer in results if gg == gender]
        if vals:
            group_wers[gender] = float(np.mean(vals))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "phrases": phrases,
            "group_wers": group_wers,
        },
        out_path,
    )

    return {
        "phrases": phrases,
        "group_wers": group_wers,
        "predictions": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/cv-valid-train.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/cv-valid-train"))
    parser.add_argument("--out", type=Path, default=Path("models/fair_asr.pt"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lambda-fair", type=float, default=0.25)
    args = parser.parse_args()

    ran_w2v2 = try_wav2vec2(args.csv, args.audio_dir)
    if ran_w2v2:
        print("Wav2Vec2 was available locally, so the demo used it for a transcription check.")
    else:
        print("Wav2Vec2 local weights were not available; using the offline phrase-ASR fallback.")

    result = train_fallback(args.csv, args.audio_dir, args.out, steps=args.steps, lambda_fair=args.lambda_fair)
    print("Final group WERs:", result["group_wers"])


if __name__ == "__main__":
    main()
