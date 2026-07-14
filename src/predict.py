import os
import sys
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BEST_MODEL_PATH, CLASS_NAMES, IDX_TO_CLASS,
    IMAGE_SIZE, MEAN, STD, MODEL_ARCH, NUM_CLASSES,
)
from src.model import build_model

# ── Load model once ────────────────────────────────────────────────────────────

_model  = None
_device = None

def _get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model  = build_model(MODEL_ARCH)
        _model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=_device))
        _model.to(_device).eval()
    return _model, _device


# ── Transform ──────────────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Predict ────────────────────────────────────────────────────────────────────

def predict_from_bytes(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _predict(image)


def predict_from_path(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    return _predict(image)


def _predict(image: Image.Image) -> dict:
    model, device = _get_model()

    tensor = _transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(probs.argmax())
    pred_key   = IDX_TO_CLASS[pred_idx]
    pred_name  = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = {
        IDX_TO_CLASS[i]: round(float(probs[i]), 4)
        for i in range(NUM_CLASSES)
    }
    # Sort by probability descending
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "predicted_class": pred_key,
        "class_name":      pred_name,
        "confidence":      round(confidence, 4),
        "probabilities":   all_probs,
    }


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json
    samples_dir = os.path.join("data", "raw", "ham10000_images")
    # pick one image per class for quick test
    from config import CLASS_LABELS
    import pandas as pd
    df = pd.read_csv(os.path.join("data", "raw", "HAM10000_metadata.csv"))

    print(f"{'True':>6}  {'Pred':>6}  {'Conf':>7}  Class Name")
    print("-" * 55)
    correct = 0
    for cls in sorted(CLASS_LABELS.keys()):
        row     = df[df["dx"] == cls].iloc[0]
        img_path = os.path.join(samples_dir, row["image_id"] + ".jpg")
        result   = predict_from_path(img_path)
        match    = "OK" if result["predicted_class"] == cls else "--"
        correct += 1 if match == "OK" else 0
        print(f"[{match}] {cls:>6}  {result['predicted_class']:>6}  {result['confidence']*100:6.1f}%  {result['class_name']}")

    print(f"\nAccuracy on 1-per-class samples: {correct}/7")
