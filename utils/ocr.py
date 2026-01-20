from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import SETTINGS


@dataclass(frozen=True)
class OCRResult:
    text: str
    conf: float


class OCRReader:
    def __init__(self):
        try:
            import easyocr
        except Exception as e:
            raise RuntimeError("easyocr is not available. Install it with: pip install easyocr") from e

        self.reader = easyocr.Reader([SETTINGS.ocr_lang], gpu=SETTINGS.ocr_gpu)
        self.allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def _preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.resize(
            gray,
            None,
            fx=float(SETTINGS.ocr_upscale),
            fy=float(SETTINGS.ocr_upscale),
            interpolation=cv2.INTER_CUBIC,
        )

        if SETTINGS.ocr_use_otsu:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return gray

    def read_plate(self, bgr_crop: np.ndarray) -> OCRResult:
        if bgr_crop is None or bgr_crop.size == 0:
            return OCRResult(text="", conf=0.0)

        img = self._preprocess(bgr_crop)

        results = self.reader.readtext(
            img,
            allowlist=self.allowlist,
            detail=1,
            paragraph=False,
        )
        if not results:
            return OCRResult(text="", conf=0.0)

        best = max(results, key=lambda r: float(r[2]))
        raw_text = str(best[1])
        conf = float(best[2])

        text = "".join(ch for ch in raw_text.strip().upper() if ch.isalnum())

        if len(text) < 4 or len(text) > 10:
            return OCRResult(text="", conf=0.0)

        return OCRResult(text=text, conf=conf)
