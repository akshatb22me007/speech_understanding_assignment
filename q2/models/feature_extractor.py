import torch.nn as nn

class SimpleSpeakerNet(nn.Module):
    def __init__(self, emb_dim=192):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(80, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)