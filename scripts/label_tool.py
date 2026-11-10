"""Review OCR drafts one crop at a time with Streamlit."""

import csv
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "data/labels/draft_labels.csv"
LABELS = ROOT / "data/labels/kaggle_labels.csv"

st.title("RiskGate plate labeling")
if not DRAFTS.exists():
    st.info("Place draft_labels.csv in data/labels/ with crop_path, group_id, draft columns.")
    st.stop()

with DRAFTS.open(newline="") as f:
    drafts = list(csv.DictReader(f))
if not {"crop_path", "group_id", "draft"}.issubset(drafts[0] if drafts else {}):
    st.error("draft_labels.csv needs crop_path, group_id, draft columns")
    st.stop()
done = set()
if LABELS.exists():
    with LABELS.open(newline="") as f:
        done = {r["crop_path"] for r in csv.DictReader(f)}
pending = [r for r in drafts if r["crop_path"] not in done]
st.write(f"Reviewed: {len(done)} / {len(drafts)}")
if not pending:
    st.success("All drafts reviewed")
    st.stop()

row = pending[0]
path = ROOT / row["crop_path"]
if not path.exists():
    st.error(f"Missing crop: {path}")
    st.stop()
st.image(str(path), caption=row["crop_path"], width=500)
with st.form("label"):
    correction = st.text_input("Plate text (leave blank to accept draft)", placeholder=row["draft"])
    accept = st.form_submit_button("Save and next")
    skip = st.form_submit_button("Skip unreadable")
if accept or skip:
    truth = "" if skip else (correction.strip() or row["draft"].strip())
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS.exists()
    with LABELS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "group_id", "truth"])
        if new_file:
            writer.writeheader()
        writer.writerow({"crop_path": row["crop_path"], "group_id": row["group_id"], "truth": truth})
    st.rerun()
