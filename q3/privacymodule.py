
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_audio(path_or_array, sr: int = 16000) -> Tuple[np.ndarray, int]:
    if isinstance(path_or_array, (str, bytes)):
        y, file_sr = librosa.load(path_or_array, sr=sr, mono=True)
        return y.astype(np.float32), file_sr
    y = np.asarray(path_or_array, dtype=np.float32)
    return y, sr


def waveform_to_mel(wav: np.ndarray, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
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
    return mel


class SpeakerEncoder(nn.Module):
    """
    Lightweight speaker encoder used to derive a biometric embedding from mel features.
    It is intentionally small so the demo can run offline.
    """

    def __init__(self, n_mels: int = 80, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T)
        x = self.net(mel)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


@dataclass
class TransformParams:
    semitones: float
    rate: float
    tilt: float


class PrivacyAnonymizer(nn.Module):
    """
    PyTorch module with two roles:
      1) speaker embedding extraction from mel spectrograms
      2) estimating a privacy-preserving voice transformation from source -> target

    The demo uses the estimated parameters for waveform-level anonymization.
    """

    def __init__(self, n_mels: int = 80, emb_dim: int = 64):
        super().__init__()
        self.encoder = SpeakerEncoder(n_mels=n_mels, emb_dim=emb_dim)

    @torch.no_grad()
    def embed_mel(self, mel: np.ndarray) -> torch.Tensor:
        mel_t = torch.from_numpy(mel).unsqueeze(0)
        emb = self.encoder(mel_t).squeeze(0)
        return emb

    @torch.no_grad()
    def embed_waveform(self, wav: np.ndarray, sr: int = 16000) -> torch.Tensor:
        mel = waveform_to_mel(wav, sr=sr)
        return self.embed_mel(mel)

    @torch.no_grad()
    def estimate_params(self, source_emb: torch.Tensor, target_emb: torch.Tensor) -> TransformParams:
        diff = target_emb - source_emb
        m = diff.mean().item()
        s = diff.std().item()

        semitones = float(np.clip(m * 9.0, -4.0, 4.0))
        rate = float(np.clip(1.0 + s * 0.05, 0.95, 1.05))
        tilt = float(np.clip(1.0 - m * 0.10, 0.85, 1.15))
        return TransformParams(semitones=semitones, rate=rate, tilt=tilt)

    def forward(self, mel: torch.Tensor, source_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T)
        diff = target_emb - source_emb
        gain = 1.0 + 0.05 * torch.tanh(diff.mean(dim=-1, keepdim=True))
        bias = 0.02 * torch.tanh(diff.std(dim=-1, keepdim=True))
        gain = gain.unsqueeze(-1)
        bias = bias.unsqueeze(-1)
        return mel * gain + bias

    @staticmethod
    def _apply_spectral_tilt(wav: np.ndarray, tilt: float) -> np.ndarray:
        # Simple pre-emphasis / de-emphasis proxy.
        # tilt > 1.0 => slightly brighter; tilt < 1.0 => slightly darker.
        coef = float(np.clip(0.95 / max(tilt, 1e-3), 0.7, 0.98))
        return librosa.effects.preemphasis(wav, coef=coef).astype(np.float32)

    @torch.no_grad()
    def anonymize_waveform(
        self,
        wav: np.ndarray,
        sr: int = 16000,
        source_ref: Optional[np.ndarray] = None,
        target_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, TransformParams]:
        source_ref = wav if source_ref is None else source_ref
        target_ref = wav if target_ref is None else target_ref

        source_emb = self.embed_waveform(source_ref, sr=sr)
        target_emb = self.embed_waveform(target_ref, sr=sr)
        params = self.estimate_params(source_emb, target_emb)

        out = wav.astype(np.float32)

        if abs(params.semitones) > 1e-3:
            out = librosa.effects.pitch_shift(out, sr=sr, n_steps=params.semitones).astype(np.float32)

        if abs(params.rate - 1.0) > 1e-3:
            out = librosa.effects.time_stretch(out, rate=params.rate).astype(np.float32)

        out = self._apply_spectral_tilt(out, params.tilt)

        # Keep content length roughly stable for ASR.
        if len(out) > len(wav):
            out = out[: len(wav)]
        elif len(out) < len(wav):
            out = np.pad(out, (0, len(wav) - len(out)))

        peak = np.max(np.abs(out)) + 1e-9
        out = 0.92 * out / peak
        return out.astype(np.float32), sr, params
