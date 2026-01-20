from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    demo_root: str = "artifacts/datasets/final_demo_data"
    out_csv: str = "artifacts/results.csv"

    detector_weights: str = "artifacts/license_plate_detector.pt"
    det_thresh: float = 0.30
    det_iou: float = 0.50
    det_device: str = "cpu"

    blur_weights: str = "utils/weights.pkl"
    blur_threshold: float = 0.55
    blur_input_h: int = 64
    blur_input_w: int = 160
    blur_device: str | None = None  # None -> auto

    ocr_lang: str = "en"
    ocr_gpu: bool = False
    ocr_min_conf: float = 0.25
    ocr_upscale: float = 2.0
    ocr_use_otsu: bool = True
    plate_min_len: int = 3
    plate_max_len: int = 12

    write_db: bool = False


SETTINGS = Settings()
