import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="ECG AI Diagnosis",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_ecg_model():
    return load_model("ecg_multilabel_1dcnn.h5")

model = load_ecg_model()

classes = ["CD", "HYP", "MI", "NORM", "STTC"]

st.title("❤️ ECG AI Diagnosis System")
st.write("AI-Based Multi-Label ECG Classification")

st.divider()

uploaded_file = st.file_uploader(
    "Upload ECG File (.npy)",
    type=["npy"]
)

if uploaded_file is not None:

    ecg = np.load(uploaded_file)

    st.success("ECG File Uploaded Successfully")

    st.subheader("ECG Signal")

    if ecg.ndim == 2:

        plot_ecg = ecg if ecg.shape[0] >= ecg.shape[1] else ecg.T

        st.line_chart(
            pd.DataFrame(
                plot_ecg,
                columns=[
                    f"Lead {i+1}"
                    for i in range(plot_ecg.shape[1])
                ]
            )
        )

    else:
        st.line_chart(ecg)

    st.divider()

    ecg_input = np.expand_dims(ecg, axis=0)

    prediction = model.predict(ecg_input)[0]

    st.subheader("AI Prediction Results")

    if len(prediction) != len(classes):
        st.error("Model output does not match the number of classes.")

    else:
        for cls, prob in zip(classes, prediction):

            percentage = float(prob) * 100

            st.write(f"**{cls}: {percentage:.2f}%**")

            st.progress(
                max(0.0, min(1.0, float(prob)))
            )

    st.divider()

    st.caption(
        "Research and educational tool. "
        "Results are not a substitute for clinical diagnosis."
    )
