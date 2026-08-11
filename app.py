import io

import pandas as pd
import streamlit as st
from PIL import Image

from rx_parser import parse_prescription

st.set_page_config(page_title="Prescription Reader", page_icon="💊", layout="wide")
st.title("💊 Prescription Reader")
st.caption("Upload a handwritten prescription → structured medication data")

model = st.selectbox("Vision model", ["qwen/qwen3.6-27b"])
uploaded = st.file_uploader("Prescription image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    data = uploaded.read()
    st.image(Image.open(io.BytesIO(data)), caption=uploaded.name, width=400)

    if st.button("Parse prescription", type="primary"):
        with st.spinner(f"{model} is reading the handwriting…"):
            try:
                rx = parse_prescription(data, model=model)
            except Exception as e:
                st.error(f"Parse failed: {e}")
                st.stop()

        # ---- header info ----
        if rx.clinic:
            st.subheader(rx.clinic)
        st.markdown(f"**Date:** {rx.date or '—'} &nbsp;|&nbsp; **Patient:** {rx.patient_name or '—'}")
        if rx.diagnoses:
            st.write(f"**Diagnoses:** {rx.diagnoses}")

        # ---- medications ----
        st.subheader("Medications")
        if rx.medications:
            df = pd.DataFrame([m.model_dump() for m in rx.medications])
            st.dataframe(df, hide_index=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode(),
                file_name="prescription.csv",
                mime="text/csv",
            )
        else:
            st.info("No medications detected.")

        with st.expander("Raw transcription / full JSON"):
            st.code(rx.raw_transcription or rx.model_dump_json(indent=2))

st.divider()
st.caption("⚠️ Research tool only — never use for actual treatment decisions.")