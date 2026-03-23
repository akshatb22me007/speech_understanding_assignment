import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
from data.voxceleb_loader import VoxCelebDataset
from models.feature_extractor import SimpleSpeakerNet
from models.disentangler import Disentangler
from tqdm import tqdm

def plot_training_curve(results_dir, epoch_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_loss.png"), dpi=200)
    plt.close()

config = yaml.safe_load(open("configs/config.yaml"))
device = config["device"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

dataset = VoxCelebDataset(config["data_path"])

train_ratio = config.get("train_split", 0.8)
train_len = int(len(dataset) * train_ratio)
test_len = len(dataset) - train_len
if test_len == 0:
    test_len = 1
    train_len = len(dataset) - 1

generator = torch.Generator().manual_seed(config.get("seed", 42))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_len, test_len], generator=generator
)

loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
feature_extractor = SimpleSpeakerNet().to(device)
disentangler = Disentangler(num_speakers=500).to(device)

optimizer = torch.optim.Adam(
    list(feature_extractor.parameters()) +
    list(disentangler.parameters()),
    lr=config["lr"]
)

ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()
epoch_losses = []

for epoch in tqdm(range(config["epochs"])):
    total_loss = 0

    for x, spk in loader:
        x, spk = x.to(device), spk.to(device)

        emb = feature_extractor(x)
        spk_code, env_code, recon, logits = disentangler(emb)

        loss = (
            mse(recon, emb) +
            ce(logits, spk)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch}: {total_loss:.4f}")

plot_training_curve(results_dir, epoch_losses)

torch.save({
    "feature_extractor": feature_extractor.state_dict(),
    "disentangler": disentangler.state_dict()
}, "results/model.pth")

# persist split indices so eval uses the same test partition
torch.save({
    "train_indices": train_dataset.indices,
    "test_indices": test_dataset.indices
}, "results/split_indices.pt")