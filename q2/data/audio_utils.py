import torchaudio
import torch

def load_audio(path, sample_rate=16000):
    waveform, sr = torchaudio.load(path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform


def extract_fbank(waveform, n_mels=80):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels
    )(waveform)

    log_mel = torch.log(mel_spec + 1e-6)
    return log_mel.squeeze(0)