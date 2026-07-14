import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HAM_IMAGES_DIR, HAM_CSV_PATH, CLASS_LABELS, NUM_CLASSES,
    IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_WORKERS,
    RANDOM_SEED, TEST_SIZE, VAL_SIZE, USE_CLASS_WEIGHTS,
)


# ── Step 2a: Data Cleaning ─────────────────────────────────────────────────────

def load_and_clean(verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(HAM_CSV_PATH)
    before = len(df)

    # Drop unknown / missing dx
    df = df[df["dx"].notna()]
    df = df[df["dx"].isin(CLASS_LABELS.keys())]

    # Remove duplicate image IDs
    df = df.drop_duplicates(subset="image_id", keep="first")

    # Verify image file exists on disk
    df["img_path"] = df["image_id"].apply(
        lambda x: os.path.join(HAM_IMAGES_DIR, f"{x}.jpg")
    )
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

    # Encode label
    df["label"] = df["dx"].map(CLASS_LABELS)

    after = len(df)

    if verbose:
        print(f"[Cleaning] Raw rows      : {before}")
        print(f"[Cleaning] After cleaning: {after}  (removed {before - after})")
        print(f"[Cleaning] Class distribution:")
        for cls, cnt in df["dx"].value_counts().items():
            print(f"           {cls:6s} -> {cnt:5d} images")

    return df


# ── Step 2b: Train / Val / Test Split ─────────────────────────────────────────

def split_dataset(df: pd.DataFrame, verbose: bool = True):
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label"], random_state=RANDOM_SEED
    )
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_df, test_size=val_relative,
        stratify=train_df["label"], random_state=RANDOM_SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    if verbose:
        print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df


# ── Step 2c: Class Weights ─────────────────────────────────────────────────────

def compute_weights(train_df: pd.DataFrame) -> torch.Tensor:
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        "balanced", classes=classes, y=train_df["label"].values
    )
    return torch.tensor(weights, dtype=torch.float32)


# ── Step 2d: Transforms ────────────────────────────────────────────────────────

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Step 2e: Dataset ───────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Step 2f: DataLoaders ───────────────────────────────────────────────────────

def get_dataloaders(verbose: bool = True):
    df                        = load_and_clean(verbose)
    train_df, val_df, test_df = split_dataset(df, verbose)

    train_ds = HAM10000Dataset(train_df, transform=get_train_transforms())
    val_ds   = HAM10000Dataset(val_df,   transform=get_val_transforms())
    test_ds  = HAM10000Dataset(test_df,  transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    class_weights = compute_weights(train_df) if USE_CLASS_WEIGHTS else None

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, weights = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print(f"[OK] Batch shape : {imgs.shape}")
    print(f"[OK] Label sample: {labels[:8].tolist()}")
    print(f"[OK] Class weights: {weights.numpy().round(3)}")
