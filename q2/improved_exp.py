import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt  
from data.voxceleb_loader import VoxCelebDataset
from models.feature_extractor import SimpleSpeakerNet
from models.disentangler import Disentangler
import torch.nn.functional as F

def binary_curves(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int32)
    scores = np.asarray(scores).astype(np.float32)

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    s_sorted = scores[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    change_idx = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[change_idx, y_sorted.size - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = s_sorted[threshold_idxs]

    # Add origin for ROC / PR plotting.
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1e-6, thresholds]

    pos = max(1, np.sum(y_true == 1))
    neg = max(1, np.sum(y_true == 0))

    tpr = tps / pos
    fpr = fps / neg
    recall = tpr.copy()

    denom = tps + fps
    precision = np.divide(tps, denom, out=np.ones_like(tps, dtype=np.float32), where=denom > 0)

    return fpr, tpr, precision, recall, thresholds


def trapz_auc(x, y):
    return float(np.trapz(y, x))


def average_precision(precision, recall):
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_roc(results_dir, fpr, tpr, roc_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "improved_roc_curve.png"), dpi=200)
    plt.close()


def plot_pr(results_dir, recall, precision, pr_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2, color="tab:orange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "improved_pr_curve.png"), dpi=200)
    plt.close()


def plot_score_distribution(results_dir, scores, y_true):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    genuine = scores[y_true == 1]
    impostor = scores[y_true == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(impostor, bins=40, alpha=0.6, label="Impostor pairs", color="tab:red", density=True)
    plt.hist(genuine, bins=40, alpha=0.6, label="Genuine pairs", color="tab:green", density=True)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title("Pairwise Score Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "improved_score_distribution.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(results_dir, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "improved_confusion_matrix.png"), dpi=200)
    plt.close()


def plot_training_curve(results_dir, epoch_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "improved_training_loss.png"), dpi=200)
    plt.close()

# =========================
# Contrastive Loss
# =========================
def contrastive_loss(embeddings, labels, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=1)

    sim_matrix = torch.matmul(embeddings, embeddings.T)

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(mask.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim_matrix / temperature) * logits_mask

    log_prob = sim_matrix / temperature - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    loss = -mean_log_prob_pos.mean()
    return loss


# =========================
# Load Config
# =========================
config = yaml.safe_load(open("configs/config.yaml"))
device = config["device"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# =========================
# Dataset + Split
# =========================
dataset = VoxCelebDataset(config["data_path"])

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(config.get("seed", 42))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# =========================
# Models
# =========================
feature_extractor = SimpleSpeakerNet().to(device)
disentangler = Disentangler(num_speakers=500).to(device)

optimizer = torch.optim.Adam(
    list(feature_extractor.parameters()) +
    list(disentangler.parameters()),
    lr=config["lr"]
)

ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()

# =========================
# TRAIN
# =========================
print("\n🚀 Training with Contrastive Loss...\n")
epoch_losses = []

for epoch in tqdm(range(config["epochs"])):
    total_loss = 0

    for x, spk in train_loader:
        x, spk = x.to(device), spk.to(device)

        emb = feature_extractor(x)
        spk_code, env_code, recon, logits = disentangler(emb)

        loss_recon = mse(recon, emb)
        loss_cls = ce(logits, spk)
        loss_contrast = contrastive_loss(spk_code, spk)

        # 🔥 Improved Loss
        loss = loss_recon + loss_cls + 0.5 * loss_contrast

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# Save model
torch.save({
    "feature_extractor": feature_extractor.state_dict(),
    "disentangler": disentangler.state_dict()
}, "results/improved_model.pth")

torch.save({
    "train_indices": train_dataset.indices,
    "test_indices": test_dataset.indices
}, "results/split_indices.pt")

plot_training_curve(results_dir, epoch_losses)


# =========================
# EVALUATION
# =========================
print("\n📊 Evaluating...\n")

feature_extractor.eval()
disentangler.eval()

embs = []
labels = []

with torch.no_grad():
    for x, spk in test_loader:
        x = x.to(device)

        emb = feature_extractor(x)
        spk_code, _, _, _ = disentangler(emb)

        embs.append(spk_code.cpu().numpy())
        labels.append(spk.item())

embs = np.vstack(embs)
labels = np.array(labels)

# Pairwise scoring
scores = []
y_true = []

for i in range(len(embs)):
    for j in range(i + 1, len(embs)):
        score = np.dot(embs[i], embs[j])
        scores.append(score)
        y_true.append(int(labels[i] == labels[j]))

y_true = np.asarray(y_true)
scores = np.asarray(scores, dtype=np.float32)

fpr, tpr, precision_curve, recall_curve, thresholds = binary_curves(y_true, scores)
fnr = 1.0 - tpr
eer_idx = int(np.argmin(np.abs(fnr - fpr)))
eer = float((fnr[eer_idx] + fpr[eer_idx]) / 2.0)
eer_threshold = float(thresholds[eer_idx])

roc_auc = trapz_auc(fpr, tpr)
pr_auc = average_precision(precision_curve, recall_curve)

y_pred = (scores >= eer_threshold).astype(np.int32)
cls_metrics = classification_metrics(y_true, y_pred)

plot_roc(results_dir, fpr, tpr, roc_auc)
plot_pr(results_dir, recall_curve, precision_curve, pr_auc)
plot_score_distribution(results_dir, scores, y_true)
plot_confusion_matrix(results_dir, cls_metrics)

print(f"\n🔥 Improved Model EER: {eer:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Accuracy@EER-threshold: {cls_metrics['accuracy']:.4f}")
print(f"F1@EER-threshold: {cls_metrics['f1']:.4f}")

# Save result
with open("results/improved_results.txt", "w") as f:
    f.write(f"EER: {eer}\n")
    f.write(f"EER_threshold: {eer_threshold}\n")
    f.write(f"ROC_AUC: {roc_auc}\n")
    f.write(f"PR_AUC: {pr_auc}\n")
    f.write(f"Accuracy_at_EER_threshold: {cls_metrics['accuracy']}\n")
    f.write(f"Precision_at_EER_threshold: {cls_metrics['precision']}\n")
    f.write(f"Recall_at_EER_threshold: {cls_metrics['recall']}\n")
    f.write(f"F1_at_EER_threshold: {cls_metrics['f1']}\n")
    f.write(f"Confusion_matrix_TN: {cls_metrics['tn']}\n")
    f.write(f"Confusion_matrix_FP: {cls_metrics['fp']}\n")
    f.write(f"Confusion_matrix_FN: {cls_metrics['fn']}\n")
    f.write(f"Confusion_matrix_TP: {cls_metrics['tp']}\n")