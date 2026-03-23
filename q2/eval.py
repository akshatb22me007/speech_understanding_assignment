import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

from data.voxceleb_loader import VoxCelebDataset
from models.feature_extractor import SimpleSpeakerNet
from models.disentangler import Disentangler


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
    plt.title("ROC Curve (Base Model)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=200)
    plt.close()


def plot_pr(results_dir, recall, precision, pr_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2, color="tab:orange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Base Model)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pr_curve.png"), dpi=200)
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
    plt.title("Pairwise Score Distribution (Base Model)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_distribution.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(results_dir, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Base Model)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

config = yaml.safe_load(open("configs/config.yaml"))
device = config["device"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

dataset = VoxCelebDataset(config["data_path"])

split_path = "results/split_indices.pt"
if os.path.exists(split_path):
    split_data = torch.load(split_path)
    test_indices = split_data.get("test_indices")
    if test_indices is None:
        test_indices = split_data
    test_dataset = Subset(dataset, test_indices)
    print(f"Loaded {len(test_indices)} test samples from saved split.")
else:
    # fallback: recreate split using config in case split file is missing
    train_ratio = config.get("train_split", 0.8)
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    if test_len == 0:
        test_len = 1
        train_len = len(dataset) - 1
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len], generator=generator
    )
    print(f"Split missing; using fallback with {len(test_dataset)} test samples.")

loader = DataLoader(test_dataset, batch_size=1)

feature_extractor = SimpleSpeakerNet().to(device)
disentangler = Disentangler(num_speakers=500).to(device)

ckpt = torch.load("results/model.pth")
feature_extractor.load_state_dict(ckpt["feature_extractor"])
disentangler.load_state_dict(ckpt["disentangler"])

feature_extractor.eval()
disentangler.eval()

embs, labels = [], []

with torch.no_grad():
    for x, spk in loader:
        x = x.to(device)
        emb = feature_extractor(x)
        spk_code, _, _, _ = disentangler(emb)

        embs.append(spk_code.cpu().numpy())
        labels.append(spk.item())

embs = np.vstack(embs)

scores, y_true = [], []

for i in range(len(embs)):
    for j in range(i+1, len(embs)):
        scores.append(np.dot(embs[i], embs[j]))
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

print("EER:", eer)
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)
print("Accuracy@EER-threshold:", cls_metrics["accuracy"])
print("F1@EER-threshold:", cls_metrics["f1"])

with open(os.path.join(results_dir, "eval_results.txt"), "w") as f:
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