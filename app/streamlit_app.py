"""Offline plate reading demo and transparent experiment dashboard."""

import csv
import hashlib
import io
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import streamlit as st

from riskgate import config
from riskgate.calibration import load, predict
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.engines.fastplate_engine import FastPlateEngine
from riskgate.features import plate_features
from riskgate.grammar import classify
from riskgate.metrics import normalize
from riskgate.router import route

st.set_page_config(page_title="RiskGate • ANPR Lab", page_icon="◈", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:#07111e;color:#e7f2f5;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#0c1b2b;border-right:1px solid #1e3547;}
[data-testid="stSidebar"] * {color:#dceef1;}
.block-container {max-width:1320px;padding-top:1.3rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.04em;}
h2 {color:#ecf8fb!important;}
.hero {position:relative;overflow:hidden;border:1px solid #254c5d;border-radius:26px;padding:36px 42px;background:radial-gradient(circle at 80% 25%,#155f64 0%,transparent 37%),linear-gradient(120deg,#102a3b,#0b1c2b 72%);margin-bottom:25px;box-shadow:0 24px 55px #02070c66;}
.hero:after {content:'◉';position:absolute;right:5%;top:-85px;font-size:300px;line-height:1;color:#59edcf10;}
.eyebrow {color:#5ef0d0;font-size:.76rem;font-weight:700;letter-spacing:.24em;text-transform:uppercase;}
.hero h1 {font-size:clamp(2.5rem,5vw,4.8rem);line-height:1.02;color:white;margin:.4rem 0 .7rem;}
.hero p {max-width:700px;color:#bdd6dc;font-size:1.08rem;line-height:1.55;margin:0;}
.chip {display:inline-block;border:1px solid #3b6d73;background:#1a4148aa;color:#91ffdf;border-radius:999px;padding:6px 12px;font-size:.76rem;font-weight:700;margin-bottom:12px;}
.section-kicker {color:#70e9d1;text-transform:uppercase;letter-spacing:.16em;font-size:.77rem;font-weight:700;margin:8px 0;}
.card {border:1px solid #224154;border-radius:18px;background:#102235;padding:19px 22px;min-height:124px;box-shadow:0 14px 30px #02091255;}
.card .label {font-size:.72rem;color:#94b6c1;letter-spacing:.12em;text-transform:uppercase;font-weight:700;}
.card .value {font-family:'Space Grotesk',sans-serif;color:#f4fcfc;font-size:2rem;font-weight:700;line-height:1.2;margin:8px 0 4px;}
.card .note {font-size:.83rem;color:#98b8c2;}
.stage {display:inline-block;border:1px solid #2c4b59;background:#132b3c;border-radius:999px;padding:8px 13px;margin:0 5px 9px 0;color:#c9edf0;font-size:.82rem;}
.stage strong {color:#6df2d4;margin-right:6px;}
.plate {font-family:'Space Grotesk',sans-serif;letter-spacing:.09em;font-weight:700;font-size:clamp(1.4rem,3vw,2.5rem);color:#10212a;background:linear-gradient(130deg,#f8f9e9,#dce9e5);border:4px solid #d9e3de;box-shadow:0 6px 0 #243c45;border-radius:9px;padding:12px 20px;display:inline-block;margin:12px 0;}
.decision {border-left:4px solid #5fe6ca;background:#12343c;border-radius:8px;padding:16px 19px;margin:10px 0;color:#e9fcf6;}
.decision.review {border-color:#ffba6b;background:#38291f;color:#fff3e4;}
.small-muted {color:#96b6c0;font-size:.9rem;line-height:1.5;}
div[data-testid="stMetric"] {background:#102235;border:1px solid #224154;border-radius:16px;padding:12px 16px;}
div[data-testid="stMetricLabel"] {color:#9bbbc5;}
div[data-testid="stMetricValue"] {color:#f0faf8;}
div[data-testid="stTabs"] button {font-family:'Space Grotesk',sans-serif;font-weight:700;}
div[data-testid="stFileUploader"] {background:#102235;border:1px dashed #3b7079;border-radius:18px;padding:8px 16px;}
div.stButton button[kind="primary"] {background:#5fe6ca;color:#08202a;border:none;font-weight:800;border-radius:10px;}
.foot {border-top:1px solid #264352;margin-top:35px;padding-top:16px;color:#789ba8;font-size:.8rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, note: str) -> None:
    st.markdown(f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div><div class="note">{escape(note)}</div></div>', unsafe_allow_html=True)


@st.cache_resource
def reader(name: str):
    return FastPlateEngine() if name == "FastPlateOCR" else EasyOCREngine(config.READER_UPSCALE, config.READER_ALLOWLIST)


@st.cache_resource
def detector(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def demo_examples() -> dict[str, dict]:
    gallery = config.ROOT / "data/sample_gallery/manifest.csv"
    if gallery.exists():
        with gallery.open(newline="") as source:
            return {f"{index:02d} · {row['truth']}": row
                    for index, row in enumerate(csv.DictReader(source), 1)}
    baseline = config.RESULTS_DIR / "end_to_end_details.csv"
    manifest = config.ROOT / "data/indian_manifest.csv"
    if not baseline.exists() or not manifest.exists():
        return {}
    with baseline.open(newline="") as source:
        rows = list(csv.DictReader(source))
    with manifest.open(newline="") as source:
        original = {r["image_path"]: r for r in csv.DictReader(source)}
    selected = [next((r for r in rows if int(r["ok"]) == outcome and r["route"] == decision
                      and r["image"] in original), None)
                for outcome, decision in ((1, "ACCEPT"), (0, "REVIEW"))]
    result = {}
    for index, item in enumerate(selected):
        if item:
            result[f"Example {index+1} · {'accepted read' if index == 0 else 'difficult read'}"] = original[item["image"]]
    return result


def analyze(image: np.ndarray, full_image: bool, model: dict | None,
            reader_name: str, target: str = "0.10") -> dict:
    marked = image.copy()
    crop = image
    detection = None
    if full_image:
        weights = config.ROOT / "weights/plate_detector.pt"
        if not weights.exists():
            return {"error": "Detector weights are not ready. Choose Plate crop mode."}
        result = detector(str(weights)).predict(image, conf=0.25, verbose=False)[0]
        if not len(result.boxes):
            return {"error": "No plate was detected. Send this image for manual review."}
        box = max(result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        cv2.rectangle(marked, (x1, y1), (x2, y2), (95, 230, 202), 3)
        detection = float(box.conf[0])
    if crop.size == 0:
        return {"error": "The detected crop was empty. Review the image manually."}
    read = reader(reader_name).read(crop)
    text = normalize(read.text)
    calibrated = predict(model, plate_features(crop, read)) if model and text else None
    if calibrated is None:
        decision = "REVIEW"
        reason = "No calibrated confidence is available" if not model else (read.error or "No readable text")
    else:
        initial = route(text, calibrated, model["thresholds"][target])
        decision = "ACCEPT" if initial == "ACCEPT" else "REVIEW"
        reason = f"Confidence passed the threshold chosen for the {float(target):.0%} validation target" if initial == "ACCEPT" else (
            "Eligible for an optional second opinion; review offline" if initial == "ESCALATE" else "Low confidence or invalid format")
    return {"marked": marked, "crop": crop, "text": text, "confidence": calibrated,
            "raw_confidence": read.raw_conf, "route": decision, "reason": reason,
            "detection": detection, "latency_ms": read.latency_ms,
            "grammar": classify(text), "error": read.error}


def save_correction(image_bytes: bytes, proposed: str, corrected: str) -> None:
    path = config.RESULTS_DIR / "review_corrections.csv"
    path.parent.mkdir(exist_ok=True)
    first = not path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["time", "image_sha256", "proposed", "corrected"])
        if first:
            writer.writeheader()
        writer.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                         "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                         "proposed": proposed, "corrected": normalize(corrected)})


model_path = config.CALIBRATION_MODEL
calibration = load(model_path) if model_path.exists() else None
selected_path = config.RESULTS_DIR / "selected_reader.json"
reader_name = json.loads(selected_path.read_text())["reader"] if selected_path.exists() else "EasyOCR"
final_path = config.RESULTS_DIR / "final_test.json"
final = json.loads(final_path.read_text()) if final_path.exists() else None
detector_path = config.ROOT / "weights/plate_detector.pt"
end_path = config.RESULTS_DIR / "end_to_end_extra.json"
if not end_path.exists():
    end_path = config.RESULTS_DIR / "end_to_end_test.json"
end = json.loads(end_path.read_text()) if end_path.exists() else None

st.sidebar.markdown("### RISK<span style='color:#5fe6ca'>GATE</span> / Console", unsafe_allow_html=True)
st.sidebar.caption("Local inference · No API calls")
policy_target = st.sidebar.selectbox("Review policy", ["0.10", "0.05", "0.02"],
                                     format_func=lambda value: f"{float(value):.0%} validation target")
st.sidebar.caption("A target guides threshold choice. It does not guarantee that error rate on new images.")
st.sidebar.divider()
st.sidebar.write("**Active components**")
st.sidebar.write(f"Detector: {'ready' if detector_path.exists() else 'unavailable'}")
st.sidebar.write(f"Reader: {reader_name}")
st.sidebar.write(f"Confidence model: {'ready' if calibration else 'unavailable'}")
st.sidebar.caption("Inputs and corrections stay on this machine.")

st.markdown("""<div class="hero"><span class="chip">OFFLINE • INDIAN ANPR RESEARCH</span>
<div class="eyebrow">Computer vision / selective prediction</div>
<h1>RISK<span style="color:#5ff0d2">GATE</span><span style="font-weight:400;color:#92adba"> / ANPR</span></h1>
<p>Read the plate. Measure uncertainty. Route the difficult cases to a human. Every decision is visible, and every result is tied to a real test.</p></div>""", unsafe_allow_html=True)

top = st.columns(4, gap="medium")
with top[0]: card("OCR labels", "1,695", "paired images and text")
with top[1]: card("Detector train", "1,027", "847 core + 180 screened")
with top[2]: card("Demo gallery", "12", "real vehicle scenes")
with top[3]:
    card("Auto coverage", f"{end['coverage']:.1%}" if end else "PENDING", "full-image evaluation")

st.markdown("<div class='section-kicker'>System flow</div>", unsafe_allow_html=True)
st.markdown("""<span class="stage"><strong>01</strong> Detect plate</span><span class="stage"><strong>02</strong> Read characters</span><span class="stage"><strong>03</strong> Check format</span><span class="stage"><strong>04</strong> Estimate correctness</span><span class="stage"><strong>05</strong> Accept or review</span>""", unsafe_allow_html=True)

demo_tab, batch_tab, evidence_tab, method_tab = st.tabs(
    ["◈   Live analysis", "▦   Batch review", "▥   Measured evidence", "◎   Method & limits"])

with demo_tab:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Analyze an image")
        st.markdown("<div class='small-muted'>Use a vehicle photo for the complete pipeline, or a plate crop to inspect the reader on its own.</div>", unsafe_allow_html=True)
        mode = st.radio("Input type", ["Full vehicle photo", "Plate crop"], horizontal=True,
                        index=0 if detector_path.exists() else 1)
        source = st.radio("Image source", ["Upload", "Demo gallery"], horizontal=True)
        image_bytes = None
        example_truth = None
        if source == "Upload":
            uploaded = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])
            if uploaded:
                image_bytes = uploaded.getvalue()
        else:
            examples = demo_examples()
            if examples:
                label = st.selectbox("Choose an example", list(examples), index=1 if len(examples) > 2 else 0)
                row = examples[label]
                path = config.ROOT / row["image_path"]
                image_bytes = path.read_bytes()
                if mode == "Plate crop":
                    if row.get("crop_path"):
                        image_bytes = (config.ROOT / row["crop_path"]).read_bytes()
                    elif all(row.get(key) for key in ("xmin", "ymin", "xmax", "ymax")):
                        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        x1, y1, x2, y2 = [int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
                        crop = image[y1:y2, x1:x2]
                        image_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
                example_truth = row["truth"]
                st.image(str(path), caption="Real vehicle scene · visually checked label · demo only", width=360)
            else:
                st.info("Demo examples are unavailable on this machine.")
        if st.button("Analyze plate", type="primary", width="stretch", disabled=image_bytes is None):
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("Could not decode this image.")
            else:
                st.session_state["analysis"] = analyze(image, mode == "Full vehicle photo", calibration,
                                                        reader_name, policy_target)
                st.session_state["source_bytes"] = image_bytes
                st.session_state["example_truth"] = example_truth
        if "analysis" in st.session_state:
            result = st.session_state["analysis"]
            if result.get("error") and "marked" not in result:
                st.warning(result["error"])
            else:
                st.image(cv2.cvtColor(result["marked"], cv2.COLOR_BGR2RGB), caption="Input and detected region", width="stretch")
    with right:
        st.subheader("Decision trace")
        if "analysis" not in st.session_state or "marked" not in st.session_state["analysis"]:
            st.markdown("<div class='card'><div class='label'>Awaiting input</div><div class='value'>Ready to analyze</div><div class='note'>Upload a photo or select a built-in example.</div></div>", unsafe_allow_html=True)
        else:
            result = st.session_state["analysis"]
            image_col, text_col = st.columns([1, 1.1], gap="medium")
            with image_col:
                st.image(cv2.cvtColor(result["crop"], cv2.COLOR_BGR2RGB), caption="Plate crop", width="stretch")
            with text_col:
                st.markdown("<div class='section-kicker'>Local OCR result</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='plate'>{escape(result['text'] or 'UNREADABLE')}</div>", unsafe_allow_html=True)
                if st.session_state.get("example_truth"):
                    st.caption(f"Example label: {st.session_state['example_truth']}")
            cols = st.columns(3)
            cols[0].metric("Correctness estimate", f"{result['confidence']:.1%}" if result["confidence"] is not None else "—")
            cols[1].metric("OCR model score", f"{result['raw_confidence']:.1%}")
            cols[2].metric("OCR time", f"{result['latency_ms']:.0f} ms")
            style = "" if result["route"] == "ACCEPT" else " review"
            st.markdown(f"<div class='decision{style}'><strong>{result['route']}</strong> · {escape(result['reason'])}<br><span class='small-muted'>Format: {escape(result['grammar'])}" +
                        (f" · Detector score: {result['detection']:.1%}" if result["detection"] is not None else "") + "</span></div>", unsafe_allow_html=True)
            with st.expander("Human correction"):
                with st.form("correction_form"):
                    corrected = st.text_input("Correct plate text", value=result["text"])
                    submitted = st.form_submit_button("Save correction")
                if submitted and normalize(corrected):
                    save_correction(st.session_state["source_bytes"], result["text"], corrected)
                    st.success("Correction saved locally in the review log.")

with batch_tab:
    st.subheader("Review a batch")
    st.caption("Analyze up to 20 images at once. Processing stays local; export the decision list as CSV. Manual review is required for uncertain or failed reads.")
    batch_mode = st.radio("Batch input", ["Full vehicle photos", "Plate crops"], horizontal=True)
    uploads = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key="batch_uploads")
    if len(uploads) > 20:
        st.error("Choose at most 20 images per batch.")
    if st.button("Analyze batch", type="primary", disabled=not uploads or len(uploads) > 20):
        records = []
        progress = st.progress(0)
        for index, uploaded in enumerate(uploads, 1):
            image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            result = analyze(image, batch_mode == "Full vehicle photos", calibration,
                             reader_name, policy_target) if image is not None else {"error": "Invalid image"}
            records.append({"file": uploaded.name, "plate": result.get("text", ""),
                            "decision": result.get("route", "REVIEW"),
                            "correctness_estimate": round(result["confidence"], 3) if result.get("confidence") is not None else "",
                            "reason": result.get("reason", result.get("error", ""))})
            progress.progress(index / len(uploads))
        st.session_state["batch_results"] = records
    if st.session_state.get("batch_results"):
        records = st.session_state["batch_results"]
        st.dataframe(records, width="stretch", hide_index=True)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        st.download_button("Export decisions as CSV", output.getvalue(), "riskgate_decisions.csv", "text/csv")
    correction_path = config.RESULTS_DIR / "review_corrections.csv"
    if correction_path.exists():
        with st.expander("Local correction log"):
            st.dataframe(list(csv.DictReader(correction_path.open(newline=""))),
                         width="stretch", hide_index=True)

with evidence_tab:
    st.subheader("What the held-out evaluation actually showed")
    st.caption("A 10% target was chosen on validation data. The final-test error can exceed that target; the interval shows uncertainty.")
    if final:
        target = final["targets"]["0.10"]
        cols = st.columns(4)
        cols[0].metric("Test plates", final["n"])
        cols[1].metric("Accepted", target["accepted"])
        cols[2].metric("Auto coverage", f"{target['coverage']:.1%}")
        cols[3].metric("Error among accepted", f"{target['observed_error']:.1%}" if target["observed_error"] is not None else "—")
        st.warning("At the stricter 2% and 5% validation targets, no test plates were accepted automatically. The 10% target accepted 72/254, and 9 of those 72 were wrong. This dataset also contains some non-Indian examples and unverified XML labels.")
        if end:
            st.subheader("Complete vehicle-photo pipeline")
            end_cols = st.columns(4)
            end_cols[0].metric("Exact plate matches", f"{end['whole_plate_exact']}/{end['n']}")
            end_cols[1].metric("End-to-end accuracy", f"{end['whole_plate_accuracy']:.1%}")
            end_cols[2].metric("Accepted automatically", end["accepted"])
            end_cols[3].metric("Accepted errors", end["accepted_errors"])
            st.caption(f"Full-image route: {end['accepted']}/{end['n']} accepted, {end['accepted_errors']} wrong ({end['accepted_error']:.1%} observed accepted error). The expanded detector was evaluated after the earlier test results were viewed.")
        plot_cols = st.columns(2, gap="large")
        for column, name, title in ((plot_cols[0], "risk_coverage.png", "Risk versus coverage"),
                                    (plot_cols[1], "reliability.png", "Confidence calibration")):
            with column:
                path = config.RESULTS_DIR / name
                if path.exists():
                    st.image(str(path), caption=title, width="stretch")
        detector_report = config.RESULTS_DIR / "detector_test.json"
        if detector_report.exists():
            report = json.loads(detector_report.read_text())
            st.subheader("Full-image detector · expanded training")
            dcols = st.columns(3)
            dcols[0].metric("Test images", report["test_images"])
            dcols[1].metric("Found a plate", report["images_with_detection"])
            dcols[2].metric("Matched box at IoU ≥ 0.5", f"{report['recall_at_iou_05']:.1%}")
            comparison = config.RESULTS_DIR / "detector_comparison.json"
            if comparison.exists():
                versions = json.loads(comparison.read_text())
                st.dataframe([{"version": "Original", "training images": 847,
                               "matched boxes": versions["original"]["matched_at_iou_05"],
                               "recall": f"{versions['original']['recall_at_iou_05']:.1%}"},
                              {"version": "Expanded", "training images": 1027,
                               "matched boxes": versions["expanded_training"]["matched_at_iou_05"],
                               "recall": f"{versions['expanded_training']['recall_at_iou_05']:.1%}"}],
                             width="stretch", hide_index=True)
        summary = config.RESULTS_DIR / "baseline_v2_summary.txt"
        if summary.exists():
            with st.expander("Complete OCR comparison"):
                st.code(summary.read_text(), language="text")
    else:
        st.info("The evaluation pipeline has not produced final numbers yet.")

with method_tab:
    st.subheader("A readable system, not a black box")
    st.markdown("""
**Detection** finds a rectangle around the plate in a full vehicle image. **OCR** reads letters and digits from that crop. **Calibration** uses labeled examples to estimate whether the whole reading is right. **Routing** accepts only readings above a threshold chosen on validation data; others go to review.

The data is split by repeated plate identity and source video. The local reader is selected using the training split; confidence is fitted on calibration data, and thresholds are chosen on validation data. An additional box-only source added 180 screened full vehicle photos to detector training. Its other 1,902 plate crops have no trustworthy text labels and are not used for OCR evaluation. The 12-gallery examples are original full vehicle photos from the first source, with their XML plate text visually checked; they are demo examples, not a benchmark. The source XML labels have only a sampled visual audit so far, and some Google images show foreign plates. The OCR test statistics were viewed before the routing evaluation, and the expanded detector was evaluated on that same test. These results are **held out from fitting but not fully blinded**.

This is an academic prototype. Glare, motion blur, tiny crops, uncommon layouts, and new cameras can reduce accuracy. Human correction remains necessary for uncertain cases. No plate image or API key is sent to a cloud service by this demo.
""")
    st.markdown("**Research basis:** [BLPR](https://arxiv.org/abs/2604.09927) · [Neural Sentinel](https://arxiv.org/abs/2602.07051) · [Nigerian VLM evaluation](https://arxiv.org/abs/2607.02025) · [VehiclePaliGemma](https://arxiv.org/abs/2412.14197) · [ICPR low-resolution competition](https://arxiv.org/abs/2604.22506) · [Indian LPR dataset](https://arxiv.org/abs/2111.06054)")

st.markdown("<div class='foot'>RISK<span style='color:#5fe6ca'>GATE</span> / ANPR LAB · Local inference · Measured uncertainty · Human review</div>", unsafe_allow_html=True)
