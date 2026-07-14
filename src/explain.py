import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BEST_MODEL_PATH, IMAGE_SIZE, MEAN, STD,
    MODEL_ARCH, GRADCAM_ALPHA,
)
from src.model import build_model
from src.predict import _get_model, _transform


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        output = self.model(tensor)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims
        weights      = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam          = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam          = F.relu(cam)

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()


def _get_gradcam():
    model, device = _get_model()
    # EfficientNet-B3: last conv block before classifier
    target_layer = model.features[-1]
    return GradCAM(model, target_layer), device


_gradcam_instance = None

def _get_gradcam_cached():
    global _gradcam_instance
    if _gradcam_instance is None:
        _gradcam_instance = _get_gradcam()
    return _gradcam_instance


# ── Heatmap overlay ────────────────────────────────────────────────────────────

def _apply_heatmap(original_img: Image.Image, cam: np.ndarray) -> Image.Image:
    # Resize CAM to image size
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            original_img.size, Image.BILINEAR
        )
    ) / 255.0

    # Colormap: blue -> green -> red
    r = np.clip(cam_resized * 2 - 1, 0, 1)
    g = np.clip(1 - np.abs(cam_resized * 2 - 1), 0, 1)
    b = np.clip(1 - cam_resized * 2, 0, 1)
    heatmap = np.stack([r, g, b], axis=2)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    # Blend with original
    orig_arr  = np.array(original_img.convert("RGB")).astype(float)
    heat_arr  = np.array(heatmap_img).astype(float)
    blended   = (orig_arr * (1 - GRADCAM_ALPHA) + heat_arr * GRADCAM_ALPHA).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── Public API ─────────────────────────────────────────────────────────────────

def explain_from_bytes(image_bytes: bytes, class_idx: int) -> str:
    """Returns base64-encoded PNG of the Grad-CAM overlay."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _explain(image, class_idx)


def explain_from_path(image_path: str, class_idx: int) -> str:
    image = Image.open(image_path).convert("RGB")
    return _explain(image, class_idx)


def _explain(image: Image.Image, class_idx: int) -> str:
    gradcam, device = _get_gradcam_cached()
    tensor = _transform(image).unsqueeze(0).to(device)
    cam    = gradcam.generate(tensor, class_idx)
    overlay = _apply_heatmap(image, cam)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from config import CLASS_LABELS, IDX_TO_CLASS, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df       = pd.read_csv(os.path.join("data", "raw", "HAM10000_metadata.csv"))
    img_dir  = os.path.join("data", "raw", "ham10000_images")

    # Test on one image per class
    for cls, idx in CLASS_LABELS.items():
        row      = df[df["dx"] == cls].iloc[0]
        img_path = os.path.join(img_dir, row["image_id"] + ".jpg")

        b64 = explain_from_path(img_path, idx)

        # Decode and save
        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{cls}.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))

        print(f"[OK] {cls:6s} -> {out_path}")

    print("\nAll Grad-CAM overlays saved to outputs/")
