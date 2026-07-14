import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PATH, BEST_MODEL_PATH, MODEL_DIR, OUTPUT_DIR,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, RANDOM_SEED, MODEL_ARCH,
)
from src.data_loader import get_dataloaders
from src.model import build_model

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [Train]", ncols=90, leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds    = outputs.max(1)
        correct    += preds.eq(labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs} [Val]  ", ncols=90, leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs    = model(imgs)
        loss       = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, preds    = outputs.max(1)
        correct    += preds.eq(labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, class_weights = get_dataloaders()

    model     = build_model(MODEL_ARCH).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining {MODEL_ARCH}  |  {EPOCHS} epochs  |  batch {train_loader.batch_size}\n")
    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>10}  {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS)
        va_loss, va_acc = validate(model, val_loader, criterion, device, epoch, EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        lr  = scheduler.get_last_lr()[0]
        tag = " <-- best" if va_acc > best_val_acc else ""
        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.4f}  {va_loss:8.4f}  {va_acc:7.4f}  {lr:10.2e}  {elapsed:5.1f}s{tag}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)

    hist_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val acc : {best_val_acc:.4f}")
    print(f"Model saved  : {BEST_MODEL_PATH}")
    print(f"History saved: {hist_path}")
    return history


if __name__ == "__main__":
    train()
