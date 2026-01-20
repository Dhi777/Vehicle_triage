from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import SETTINGS


class TinyBlurCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


@dataclass(frozen=True)
class BlurResult:
    p_blur: float
    threshold: float
    is_blurry: bool


class BlurPredictor:
    def __init__(self) -> None:
        self.weights_path = Path(SETTINGS.blur_weights)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Blur CNN weights not found: {self.weights_path.resolve()}")

        if SETTINGS.blur_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = SETTINGS.blur_device

        self.input_size: Tuple[int, int] = (int(SETTINGS.blur_input_h), int(SETTINGS.blur_input_w))
        self.threshold: float = float(SETTINGS.blur_threshold)

        self.model = TinyBlurCNN().to(self.device).eval()

        ckpt = torch.load(str(self.weights_path), map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            if "blur_threshold" in ckpt:
                self.threshold = float(ckpt["blur_threshold"])
        elif isinstance(ckpt, dict):
            self.model.load_state_dict(ckpt)
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    def _to_tensor(self, bgr_crop: np.ndarray) -> torch.Tensor:
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("Empty crop passed to BlurPredictor")

        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA)

        x = torch.from_numpy(gray).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0)
        x = (x - 0.5) / 0.5
        return x.to(self.device)

    @torch.no_grad()
    def predict(self, bgr_crop: np.ndarray) -> BlurResult:
        x = self._to_tensor(bgr_crop)
        logits = self.model(x)
        p = float(torch.sigmoid(logits).item())
        t = float(self.threshold)
        return BlurResult(p_blur=p, threshold=t, is_blurry=(p >= t))


_BLUR_PREDICTOR: Optional[BlurPredictor] = None


def get_blur_predictor() -> BlurPredictor:
    global _BLUR_PREDICTOR
    if _BLUR_PREDICTOR is None:
        _BLUR_PREDICTOR = BlurPredictor()
    return _BLUR_PREDICTOR


def run_blur_gate(bgr_crop: np.ndarray) -> BlurResult:
    return get_blur_predictor().predict(bgr_crop)
