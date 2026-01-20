# Vehicle Triage Prototype

This repository implements a small, explainable License Plate detection triage pipeline for vehicle images.

Each image ends in exactly one outcome:
- 'AUTO_OCR_SUCCESS'
- 'MANUAL_REVIEW_REQUIRED' with a reason ('NO_PLATE', 'BLURRY', 'LOW_OCR_CONF')

## Pipeline
1. Plate Detection: YOLO plate detector produces a bounding box and confidence.
2. Blur Quality Gate: a tiny CNN scores blur probability on a tight plate crop.
3. OCR: EasyOCR extracts text and confidence from a padded plate crop.
4. Routing: deterministic rules produce the final outcome and reason.
5. Outputs: results are written to 'artifacts/results.csv' (optional DB persistence).

## Project Folder Layout
- 'main.py' — batch pipeline runner (writes CSV, optional DB)
- 'app.py' — Streamlit dashboard to run and inspect results
- 'config.py' — hardcoded settings used across the project
- 'utils/cnn.py' — blur model + predictor (cached in-process)
- 'utils/ocr.py' — EasyOCR wrapper + preprocessing
- 'utils/database.py' — optional PostgreSQL persistence
- 'training/train_blur_cnn.py' — training script for the blur CNN (not required for inference)
- 'artifacts/' — assets (detector weights, datasets). `results.csv` is generated.

## How to Run

### 1) Setup
Create and activate a virtual environment, then install dependencies:

$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt


### 2) Run the batch pipeline:

$ python3 main.py

- This produces:
artifacts/results.csv


### 3) Run the dashboard

$ streamlit run app.py


## Configuration
All settings are hardcoded in config.py via a single SETTINGS object (paths, thresholds, OCR settings, DB toggle).

##Output Schema (CSV)
Each row contains:
image_path, split, variant, det_conf, plate_bbox, p_blur, blur_threshold, blur_decision, ocr_text, ocr_conf, final_outcome, manual_reason
