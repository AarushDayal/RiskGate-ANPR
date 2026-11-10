"""Cropped-plate demo; model and cache are produced by the evaluation pipeline."""

import cv2
import numpy as np
import streamlit as st

from riskgate import config
from riskgate.calibration import load
from riskgate.cascade import process
from riskgate.engines.easyocr_engine import EasyOCREngine
from riskgate.engines.gemini_engine import GeminiEngine
from riskgate.grammar import classify

st.set_page_config(page_title="RiskGate ANPR")
st.title("RiskGate ANPR")
demo, evidence = st.tabs(["Read a plate", "Risk and coverage"])

with demo:
    st.caption("Upload an already-cropped plate. Full-image detection awaits detector training.")
    uploaded = st.file_uploader("Plate crop", type=["jpg", "jpeg", "png"])
    offline = st.checkbox("Offline demo mode (cached Gemini answers only)", value=True)
    run = st.checkbox("Allow real Gemini calls on ESCALATE", value=False, disabled=offline)
    target = st.selectbox("Target error rate", ["0.02", "0.05", "0.10"], index=1)
    if not (config.ROOT / "calibration.json").exists():
        st.warning("Calibration is not ready. Review at least 200 plates, run baseline_v2, then fit the router.")
    elif uploaded and st.button("Read plate"):
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not read image")
        else:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded crop", width=500)
            model = load(config.ROOT / "calibration.json")
            reader = EasyOCREngine(config.READER_UPSCALE, config.READER_ALLOWLIST)
            gemini = GeminiEngine(run=run, offline=offline)
            decision = process(image, reader, gemini, model, target)
            st.metric("Final text", decision.final_text or "Unreadable")
            st.metric("Calibrated confidence", f"{decision.confidence:.1%}")
            st.write({"grammar": classify(decision.final_text), "route": decision.route,
                      "reason": decision.reason, "engines": decision.engines_used,
                      "Gemini calls": decision.gemini_calls_used})

with evidence:
    plot = config.RESULTS_DIR / "risk_coverage.png"
    if plot.exists():
        st.image(str(plot), caption="Final test risk and coverage")
    else:
        st.info("The protected final test has not been run yet.")
