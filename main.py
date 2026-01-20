from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import SETTINGS
from utils.cnn import run_blur_gate
from utils.database import PostgresDB
from utils.ocr import OCRReader

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator.*",
    category=UserWarning,
)

_PLATE_MODEL: Optional[YOLO] = None


def get_plate_model(weights_path: str) -> YOLO:
    global _PLATE_MODEL
    if _PLATE_MODEL is not None:
        return _PLATE_MODEL

    w = Path(weights_path)
    if not w.exists():
        raise FileNotFoundError(f"Plate detector weights not found: {w.resolve()}")

    _PLATE_MODEL = YOLO(str(w))
    return _PLATE_MODEL


def detect_plate(
    bgr: np.ndarray,
    *,
    model_weights: str,
    conf_thresh: float,
    iou_thresh: float,
    device: str,
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    model = get_plate_model(model_weights)
    results = model.predict(
        source=bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
        device=device,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, 0.0

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx].item())

    x1, y1, x2, y2 = map(int, xyxy.tolist())
    return (x1, y1, x2, y2), conf


def crop_xyxy(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    pad_px: int,
) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1 - pad_px, w - 1))
    y1 = max(0, min(y1 - pad_px, h - 1))
    x2 = max(0, min(x2 + pad_px, w))
    y2 = max(0, min(y2 + pad_px, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def iter_demo_images(root: Path) -> Iterator[Tuple[str, str, Path]]:
    curated = root / "clear"
    if curated.exists():
        for variant in ("clear", "plate_blurred"):
            d = root / variant
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jpg")):
                yield "curated", variant, p
        return

    for split in ("train", "valid"):
        for variant in ("clear", "plate_blurred"):
            d = root / split / variant
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jpg")):
                yield split, variant, p


@dataclass(frozen=True)
class ImageDecision:
    final_outcome: str
    manual_reason: Optional[str]
    p_blur: Optional[float]
    blur_threshold: Optional[float]
    blur_decision: Optional[bool]
    ocr_text: str
    ocr_conf: Optional[float]


def decide_for_image(img: np.ndarray, *, ocr: OCRReader) -> Tuple[Optional[Tuple[int, int, int, int]], float, ImageDecision]:
    bbox, det_conf = detect_plate(
        img,
        model_weights=SETTINGS.detector_weights,
        conf_thresh=SETTINGS.det_thresh,
        iou_thresh=SETTINGS.det_iou,
        device=SETTINGS.det_device,
    )

    if bbox is None:
        return None, 0.0, ImageDecision(
            final_outcome="MANUAL_REVIEW_REQUIRED",
            manual_reason="NO_PLATE",
            p_blur=None,
            blur_threshold=None,
            blur_decision=None,
            ocr_text="",
            ocr_conf=None,
        )

    blur_crop = crop_xyxy(img, bbox, pad_px=0)
    if blur_crop is None:
        return bbox, det_conf, ImageDecision(
            final_outcome="MANUAL_REVIEW_REQUIRED",
            manual_reason="NO_PLATE",
            p_blur=None,
            blur_threshold=None,
            blur_decision=None,
            ocr_text="",
            ocr_conf=None,
        )

    blur_res = run_blur_gate(blur_crop)
    if blur_res.is_blurry:
        return bbox, det_conf, ImageDecision(
            final_outcome="MANUAL_REVIEW_REQUIRED",
            manual_reason="BLURRY",
            p_blur=float(blur_res.p_blur),
            blur_threshold=float(blur_res.threshold),
            blur_decision=True,
            ocr_text="",
            ocr_conf=None,
        )

    ocr_crop = crop_xyxy(img, bbox, pad_px=8)
    if ocr_crop is None:
        return bbox, det_conf, ImageDecision(
            final_outcome="MANUAL_REVIEW_REQUIRED",
            manual_reason="NO_PLATE",
            p_blur=float(blur_res.p_blur),
            blur_threshold=float(blur_res.threshold),
            blur_decision=False,
            ocr_text="",
            ocr_conf=None,
        )

    o = ocr.read_plate(ocr_crop)
    if o.conf < SETTINGS.ocr_min_conf or not o.text:
        return bbox, det_conf, ImageDecision(
            final_outcome="MANUAL_REVIEW_REQUIRED",
            manual_reason="LOW_OCR_CONF",
            p_blur=float(blur_res.p_blur),
            blur_threshold=float(blur_res.threshold),
            blur_decision=False,
            ocr_text=o.text,
            ocr_conf=float(o.conf),
        )

    return bbox, det_conf, ImageDecision(
        final_outcome="AUTO_OCR_SUCCESS",
        manual_reason=None,
        p_blur=float(blur_res.p_blur),
        blur_threshold=float(blur_res.threshold),
        blur_decision=False,
        ocr_text=o.text,
        ocr_conf=float(o.conf),
    )


def process_one(img_path: Path, *, split: str, variant: str, ocr: OCRReader) -> Optional[dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    bbox, det_conf, decision = decide_for_image(img, ocr=ocr)

    return {
        "image_path": str(img_path),
        "split": split,
        "variant": variant,
        "det_conf": round(float(det_conf), 6),
        "plate_bbox": list(bbox) if bbox is not None else None,
        "p_blur": round(float(decision.p_blur), 6) if decision.p_blur is not None else None,
        "blur_threshold": round(float(decision.blur_threshold), 3) if decision.blur_threshold is not None else None,
        "blur_decision": decision.blur_decision,
        "ocr_text": decision.ocr_text,
        "ocr_conf": round(float(decision.ocr_conf), 6) if decision.ocr_conf is not None else None,
        "final_outcome": decision.final_outcome,
        "manual_reason": decision.manual_reason,
    }


def main() -> None:
    demo_root = Path(SETTINGS.demo_root)
    ocr = OCRReader()

    db: Optional[PostgresDB] = None
    if SETTINGS.write_db:
        db = PostgresDB()
        db.create_table()

    out_csv = Path(SETTINGS.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "split", "variant",
        "det_conf", "plate_bbox",
        "p_blur", "blur_threshold", "blur_decision",
        "ocr_text", "ocr_conf",
        "final_outcome", "manual_reason",
    ]

    rows: list[dict] = []
    for split, variant, img_path in iter_demo_images(demo_root):
        row = process_one(img_path, split=split, variant=variant, ocr=ocr)
        if row is None:
            continue

        rows.append(row)
        if db is not None:
            db.insert_result(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if db is not None:
        db.close()

    print(f"Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
