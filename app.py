from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from config import SETTINGS

RESULTS_CSV = Path(SETTINGS.out_csv)


def run_pipeline() -> int:
    return subprocess.call([sys.executable, "main.py"])


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_CSV)


st.set_page_config(page_title="Vehicle Triage Dashboard", layout="wide")
st.title("Vehicle Triage Dashboard")

with st.sidebar:
    st.header("Run")
    st.caption(f"Dataset: {SETTINGS.demo_root}")
    st.caption(f"Detector conf: {SETTINGS.det_thresh}")
    st.caption(f"Blur threshold: {SETTINGS.blur_threshold}")
    st.caption(f"OCR min conf: {SETTINGS.ocr_min_conf}")
    st.caption(f"Write DB: {SETTINGS.write_db}")

    if st.button("Run pipeline", type="primary"):
        rc = run_pipeline()
        if rc == 0:
            st.success("Pipeline run completed.")
        else:
            st.error(f"Pipeline exited with code {rc}.")


df = load_results()
if df.empty:
    st.info(f"No results found yet. Click 'Run pipeline' to generate {RESULTS_CSV}.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Final outcomes")
    st.write(df["final_outcome"].value_counts(dropna=False))

with col2:
    st.subheader("Manual reasons")
    if "manual_reason" in df.columns:
        st.write(df["manual_reason"].fillna("").value_counts())

st.divider()

st.subheader("Filter")
variants = sorted(df["variant"].dropna().unique())
outcomes = sorted(df["final_outcome"].dropna().unique())
reasons = sorted(df["manual_reason"].fillna("").unique())

f_variant = st.multiselect("variant", variants, default=list(variants))
f_outcome = st.multiselect("final_outcome", outcomes, default=list(outcomes))
f_reason = st.multiselect("manual_reason", reasons, default=list(reasons))

fdf = df[
    df["variant"].isin(f_variant)
    & df["final_outcome"].isin(f_outcome)
    & df["manual_reason"].fillna("").isin(f_reason)
].copy()

st.write(f"Rows: {len(fdf)}")
st.dataframe(fdf, use_container_width=True)

st.divider()

st.subheader("Preview")
idx = st.number_input("Row index to preview (from filtered table)", min_value=0, max_value=max(len(fdf) - 1, 0), value=0)
if len(fdf) > 0:
    row = fdf.iloc[int(idx)]
    img_path = Path(row["image_path"])
    if img_path.exists():
        st.image(str(img_path), caption=str(img_path), use_container_width=True)
    else:
        st.warning(f"Image not found: {img_path}")
