import sys
import os
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, DROPOUT


def build_model(arch: str = "efficientnet_b3") -> nn.Module:
    if arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, NUM_CLASSES),
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return model


if __name__ == "__main__":
    import torch
    from config import MODEL_ARCH
    m = build_model(MODEL_ARCH)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print(f"[OK] {MODEL_ARCH} output shape: {out.shape}")
    total = sum(p.numel() for p in m.parameters())
    print(f"[OK] Total params: {total:,}")
