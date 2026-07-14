import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, CLASS_LABELS, CLASS_NAMES, IDX_TO_CLASS
from src.data_loader import load_and_clean, split_dataset, compute_weights

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]


def plot_class_distribution(df: pd.DataFrame):
    counts = df["dx"].value_counts()
    labels = [f"{k}\n({v})" for k, v in counts.items()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HAM10000 — Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    axes[0].bar(range(len(counts)), counts.values, color=COLORS, edgecolor="white")
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=30, ha="right")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Absolute counts")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=9)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=COLORS,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 8},
    )
    axes[1].set_title("Proportion")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {path}")


def plot_sample_images(df: pd.DataFrame, n_per_class: int = 3):
    classes = list(CLASS_LABELS.keys())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 3, len(classes) * 3))
    fig.suptitle("HAM10000 — Sample Images per Class", fontsize=13, fontweight="bold")

    for row_i, cls in enumerate(classes):
        subset = df[df["dx"] == cls].sample(
            n=min(n_per_class, len(df[df["dx"] == cls])), random_state=42
        )
        for col_i in range(n_per_class):
            ax = axes[row_i][col_i]
            ax.axis("off")
            if col_i < len(subset):
                img = Image.open(subset.iloc[col_i]["img_path"]).convert("RGB")
                ax.imshow(img)
                if col_i == 0:
                    ax.set_ylabel(cls, fontsize=10, fontweight="bold", rotation=0,
                                  labelpad=60, va="center")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sample_images.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {path}")


def plot_class_weights(weights):
    labels = [IDX_TO_CLASS[i] for i in range(len(weights))]
    vals   = weights.numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, vals, color=COLORS, edgecolor="white")
    ax.set_xlabel("Weight")
    ax.set_title("Class Weights (used in CrossEntropyLoss)", fontweight="bold")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="weight=1")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {path}")


def plot_split_distribution(train_df, val_df, test_df):
    splits = {"Train": train_df, "Val": val_df, "Test": test_df}
    classes = list(CLASS_LABELS.keys())
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (split_name, sdf) in enumerate(splits.items()):
        counts = [len(sdf[sdf["dx"] == c]) for c in classes]
        ax.bar(x + i * width, counts, width, label=split_name, color=COLORS[i], edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Stratified Split — Class Distribution per Subset", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "split_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved: {path}")


def save_eda_summary(df, train_df, val_df, test_df, weights):
    summary = {
        "total_images": len(df),
        "num_classes": len(CLASS_LABELS),
        "class_counts": df["dx"].value_counts().to_dict(),
        "split": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        },
        "class_weights": {
            IDX_TO_CLASS[i]: round(float(weights[i]), 4)
            for i in range(len(weights))
        },
        "image_size": "variable (resized to 224x224 during training)",
    }
    path = os.path.join(OUTPUT_DIR, "eda_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EDA] Saved: {path}")
    return summary


def run_eda():
    print("=" * 50)
    print("  EDA — HAM10000 Skin Lesion Dataset")
    print("=" * 50)

    df                        = load_and_clean(verbose=True)
    train_df, val_df, test_df = split_dataset(df, verbose=True)
    weights                   = compute_weights(train_df)

    print("\n[EDA] Generating plots...")
    plot_class_distribution(df)
    plot_sample_images(df)
    plot_class_weights(weights)
    plot_split_distribution(train_df, val_df, test_df)
    summary = save_eda_summary(df, train_df, val_df, test_df, weights)

    print("\n[EDA] Summary:")
    print(f"  Total images : {summary['total_images']}")
    print(f"  Classes      : {summary['num_classes']}")
    print(f"  Train/Val/Test: {summary['split']['train']} / {summary['split']['val']} / {summary['split']['test']}")
    print(f"  Class weights: { {k: v for k, v in summary['class_weights'].items()} }")
    print("\n[EDA] All outputs saved to:", OUTPUT_DIR)
    print("=" * 50)

    return summary


if __name__ == "__main__":
    run_eda()
