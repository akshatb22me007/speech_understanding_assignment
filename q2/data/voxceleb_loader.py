import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torch

import soundfile as sf
import torch

def load_audio(path, sample_rate=16000):
    waveform, sr = sf.read(path)

    waveform = torch.tensor(waveform).float()

    # convert to mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)

    # resample if needed
    if sr != sample_rate:
        import torchaudio
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

class VoxCelebDataset(Dataset):
    def __init__(self, root_dir):
        file_label_pairs = []

        speakers = sorted(os.listdir(root_dir))

        for spk_id, spk in enumerate(speakers):
            spk_path = os.path.join(root_dir, spk)

            for root, _, files in os.walk(spk_path):
                for file in files:
                    if file.endswith(".flac") or file.endswith(".wav"):
                        file_label_pairs.append((os.path.join(root, file), spk_id))

        # ensure deterministic ordering for reproducible splits
        file_label_pairs = sorted(file_label_pairs, key=lambda x: x[0])
        self.files = [p for p, _ in file_label_pairs]
        self.labels = [lbl for _, lbl in file_label_pairs]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        waveform = load_audio(path)
        features = extract_fbank(waveform)

        # truncate/pad
        max_len = 300
        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            pad = max_len - features.shape[1]
            features = torch.nn.functional.pad(features, (0, pad))

        return features, label