
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def load_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"filename", "text", "up_votes", "down_votes", "age", "gender", "accent"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def audit(df: pd.DataFrame) -> dict:
    valid = df[df["up_votes"] > df["down_votes"]].copy()

    metrics = {
        "n_rows": int(len(valid)),
        "missing_gender": float(valid["gender"].isna().mean()),
        "missing_age": float(valid["age"].isna().mean()),
        "missing_accent": float(valid["accent"].isna().mean()),
        "gender_distribution": valid["gender"].value_counts(dropna=False).to_dict(),
        "age_distribution": valid["age"].value_counts(dropna=False).to_dict(),
        "accent_distribution": valid["accent"].value_counts(dropna=False).to_dict(),
        "confidence_mean": float(
            (valid["up_votes"] / (valid["up_votes"] + valid["down_votes"] + 1)).mean()
        ),
        "low_confidence_share": float(
            (
                valid["up_votes"] / (valid["up_votes"] + valid["down_votes"] + 1) < 0.6
            ).mean()
        ),
    }

    gender_counts = valid["gender"].value_counts(dropna=False)
    if len(gender_counts) >= 2:
        nonzero = gender_counts[gender_counts > 0]
        metrics["gender_imbalance_ratio"] = float(nonzero.max() / nonzero.min())
    else:
        metrics["gender_imbalance_ratio"] = float("nan")

    return metrics, valid


def save_plots(valid: pd.DataFrame, out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        # Page 1: distributions
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Common Voice Audit - Representation and Documentation Debt", fontsize=15, weight="bold")

        valid["gender"].value_counts(dropna=False).sort_index().plot(
            kind="bar", ax=axes[0, 0], title="Gender distribution"
        )
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Count")

        valid["age"].value_counts(dropna=False).sort_index().plot(
            kind="bar", ax=axes[0, 1], title="Age distribution"
        )
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Count")

        valid["accent"].value_counts(dropna=False).head(10).sort_values(ascending=False).plot(
            kind="bar", ax=axes[1, 0], title="Top accents"
        )
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Count")

        conf = valid["up_votes"] / (valid["up_votes"] + valid["down_votes"] + 1)
        axes[1, 1].hist(conf, bins=10)
        axes[1, 1].set_title("Validation confidence")
        axes[1, 1].set_xlabel("Confidence")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: documentation debt summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        table_data = [
            ["Rows used", f"{len(valid)}"],
            ["Missing gender", f"{valid['gender'].isna().mean():.1%}"],
            ["Missing age", f"{valid['age'].isna().mean():.1%}"],
            ["Missing accent", f"{valid['accent'].isna().mean():.1%}"],
            ["Mean confidence", f"{conf.mean():.3f}"],
            ["Low-confidence share", f"{(conf < 0.6).mean():.1%}"],
        ]
        tbl = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.45, 0.25],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.6)
        ax.set_title("Documentation debt summary", fontsize=15, weight="bold", pad=18)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/cv-valid-train.csv"))
    parser.add_argument("--out", type=Path, default=Path("audit_plots.pdf"))
    parser.add_argument("--metrics-out", type=Path, default=Path("audit_metrics.json"))
    args = parser.parse_args()

    df = load_table(args.csv)
    metrics, valid = audit(df)

    print("=== Audit summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    save_plots(valid, args.out)

    args.metrics_out.write_text(pd.Series(metrics).to_json(indent=2))
    print(f"Saved plots -> {args.out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
