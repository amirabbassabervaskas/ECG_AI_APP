import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("ecg_multilabel_1dcnn.h5")

classes = [
    "CD",
    "HYP",
    "MI",
    "NORM",
    "STTC"
]

st.title("ECG AI Diagnosis System")

uploaded_file = st.file_uploader(
    "Upload ECG File (.npy)",
    type=["npy"]
)

if uploaded_file is not None:

    ecg = np.load(uploaded_file)

    ecg = np.expand_dims(ecg, axis=0)

    prediction = model.predict(ecg)[0]

    st.subheader("Prediction")

    for cls, prob in zip(classes, prediction):
        st.write(f"{cls}: {prob:.3f}")
