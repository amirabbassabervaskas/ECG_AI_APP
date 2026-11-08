import streamlit as st import numpy as np import matplotlib.pyplot as plt import pandas as pd from tensorflow.keras.models import load_model
st.set_page_config( page_title="ECG AI Diagnosis System", page_icon="❤️", layout="wide" )
@st.cache_resource def load_ecg_model(): return load_model("ecg_multilabel_1dcnn.h5", compile=False)
try: model = load_ecg_model() model_loaded = True except Exception as e: model_loaded = False st.error(f"AI Model Loading Error: {e}")
lead_names = [ "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6" ]
classes = ["CD", "HYP", "MI", "NORM", "STTC"]
st.title("❤️ ECG AI Diagnosis System")
st.write( "Artificial Intelligence-Based " "Multi-Label ECG Classification" )
st.info( "Upload a 12-lead ECG signal in NPY format " "to analyze the ECG and generate AI predictions." )
uploaded_file = st.file_uploader( "Upload ECG File (.npy)", type=["npy"] )
if uploaded_file is not None and model_loaded: try: ecg = np.load(uploaded_file, allow_pickle=False)
    st.subheader("ECG File Information")
    st.write("ECG Shape:", ecg.shape)

    if ecg.shape != (1000, 12):
        st.error("Invalid ECG shape. Expected shape: (1000, 12)")
        st.stop()

    if not np.issubdtype(ecg.dtype, np.number):
        st.error("ECG file must contain numerical values.")
        st.stop()

    if not np.all(np.isfinite(ecg)):
        st.error("ECG file contains invalid values.")
        st.stop()

    ecg = ecg.astype(np.float32)

    st.subheader("12-Lead ECG Visualization")

    fig, axes = plt.subplots(12, 1, figsize=(15, 22))

    for i in range(12):
        axes[i].plot(ecg[:, i], linewidth=0.8)
        axes[i].set_title(
            lead_names[i],
            loc="left",
            fontsize=10,
            fontweight="bold"
        )
        axes[i].grid(True, alpha=0.4)
        axes[i].set_xlim(0, 999)
        axes[i].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("AI Diagnosis Results")

    input_ecg = np.expand_dims(ecg, axis=0)

    prediction = model.predict(input_ecg, verbose=0)[0]

    if len(prediction) != len(classes):
        st.error("Model output does not match the five diagnosis classes.")
        st.stop()

    for cls, prob in zip(classes, prediction):
        percentage = float(prob) * 100

        st.write(f"**{cls}: {percentage:.2f}%**")

        st.progress(
            max(0.0, min(float(prob), 1.0))
        )

    st.subheader("Prediction Summary")

    results = pd.DataFrame({
        "Diagnosis Class": classes,
        "AI Score (%)": [
            round(float(p) * 100, 2)
            for p in prediction
        ]
    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Research and educational use only. "
        "AI predictions are not a substitute "
        "for professional medical diagnosis."
    )

except Exception as e:
    st.error(f"Error during ECG analysis: {e}")
