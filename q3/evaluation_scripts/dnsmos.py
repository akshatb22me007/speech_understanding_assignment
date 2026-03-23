
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    _HAVE_ONNX = True
except Exception:
    _HAVE_ONNX = False


def _proxy_dnsmos(wav: np.ndarray, sr: int = 16000) -> dict:
    wav = wav.astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-9)

    rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
    clip_ratio = float(np.mean(np.abs(wav) > 0.98))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav)))) / 2.0)

    spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=160))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=spec)))

    # Map simple audio properties into a 1-5 style MOS proxy.
    sig = 4.6 - 2.0 * clip_ratio - 0.7 * max(0.0, 0.12 - rms)
    bak = 4.4 - 2.2 * flatness - 0.4 * zcr
    ovrl = max(1.0, min(5.0, 0.55 * sig + 0.45 * bak))
    sig = float(np.clip(sig, 1.0, 5.0))
    bak = float(np.clip(bak, 1.0, 5.0))
    ovrl = float(np.clip(ovrl, 1.0, 5.0))
    return {"SIG": sig, "BAK": bak, "OVRL": ovrl, "mode": "proxy"}


class DNSMOS:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.session = None
        if _HAVE_ONNX and model_path is not None and model_path.exists():
            try:
                self.session = ort.InferenceSession(str(model_path))
            except Exception:
                self.session = None

    def compute(self, audio_path: Path) -> dict:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        wav = wav.astype(np.float32)

        if self.session is None:
            return _proxy_dnsmos(wav, sr=sr)

        # Real DNSMOS inference path (only if the ONNX model is present and the runtime works).
        audio = wav[:16000]
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        input_data = audio.astype(np.float32)[None, :]
        out = self.session.run(None, {"input_1": input_data})
        return {
            "SIG": float(out[0][0][0]),
            "BAK": float(out[0][0][1]),
            "OVRL": float(out[0][0][2]),
            "mode": "onnx",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="*", default=[Path("../examples/original.wav"), Path("../examples/anonymized.wav")])
    parser.add_argument("--model", type=Path, default=Path("../models/sig_bak_ovr.onnx"))
    args = parser.parse_args()

    dnsmos = DNSMOS(args.model)
    for p in args.audio:
        score = dnsmos.compute(p)
        print(f"{p}: {score}")


if __name__ == "__main__":
    main()
