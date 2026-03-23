import torch
import torch.nn as nn

class Disentangler(nn.Module):
    def __init__(self, input_dim=192, spk_dim=128, env_dim=128, num_speakers=100):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )

        self.spk_head = nn.Linear(256, spk_dim)
        self.env_head = nn.Linear(256, env_dim)

        self.decoder = nn.Linear(spk_dim + env_dim, input_dim)
        self.classifier = nn.Linear(spk_dim, num_speakers)

    def forward(self, x):
        z = self.encoder(x)

        spk = self.spk_head(z)
        env = self.env_head(z)

        recon = self.decoder(torch.cat([spk, env], dim=-1))
        logits = self.classifier(spk)

        return spk, env, recon, logits