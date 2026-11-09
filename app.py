import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="ECG AI Diagnosis System",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1>ECG AI Diagnosis System</h1>
    <h3>Developed by Amirabbas Sabervaskas</h3>
    <p><b>Medical Student | AI Researcher</b></p>
    <p>Research Area: Artificial Intelligence in Cardiology</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Multi-Label 12-Lead ECG Classification")

st.write(
    "Upload a 12-lead ECG signal or ECG image "
    "for artificial intelligence research analysis."
)

classes = ["CD", "HYP", "MI", "NORM", "STTC"]

lead_names = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]


@st.cache_resource
def load_ecg_model():
    return load_model(
        "ecg_multilabel_1dcnn.h5",
        compile=False
    )


try:
    model = load_ecg_model()
    st.success("AI Model Loaded Successfully")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()


def analyze_ecg_signal(ecg):

    if ecg.shape != (1000, 12):
        st.error(
            f"Incorrect ECG shape: {ecg.shape}. "
            "Expected (1000, 12)."
        )
        return

    if not np.issubdtype(ecg.dtype, np.number):
        st.error("ECG data must be numeric.")
        return

    if not np.all(np.isfinite(ecg)):
        st.error("ECG contains invalid values.")
        return

    ecg = ecg.astype(np.float32)

    st.success("ECG Signal Loaded Successfully")

    st.write("ECG Shape:", ecg.shape)

    st.subheader("12-Lead ECG Visualization")

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(15, 22)
    )

    for i in range(12):
        axes[i].plot(
            ecg[:, i],
            linewidth=0.8
        )

        axes[i].set_title(
            lead_names[i],
            loc="left"
        )

        axes[i].grid(
            True,
            alpha=0.4
        )

        axes[i].set_xlim(0, 999)

    axes[-1].set_xlabel("Sample")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("AI Diagnosis Results")

    input_ecg = np.expand_dims(
        ecg,
        axis=0
    )

    prediction = model.predict(
        input_ecg,
        verbose=0
    )[0]

    if len(prediction) != 5:
        st.error("Unexpected model output.")
        return

    results = []

    for cls, prob in zip(classes, prediction):
        percentage = float(prob) * 100

        st.write(
            f"**{cls}: {percentage:.2f}%**"
        )

        st.progress(
            float(np.clip(prob, 0, 1))
        )

        results.append({
            "Diagnosis Class": cls,
            "AI Score (%)": round(percentage, 2)
        })

    st.subheader("Prediction Summary")

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Research and educational use only. "
        "AI scores are not a substitute for professional diagnosis."
    )


st.markdown("---")
st.header("ECG Signal Analysis")

uploaded_file = st.file_uploader(
    "Upload ECG Signal (.npy)",
    type=["npy"],
    key="npy_upload"
)

if uploaded_file is not None:
    try:
        ecg = np.load(
            uploaded_file,
            allow_pickle=False
        )

        analyze_ecg_signal(ecg)

    except Exception as e:
        st.error(f"ECG Analysis Error: {e}")


st.markdown("---")
st.header("ECG Image Analysis")

st.write(
    "Upload a standard 12-lead ECG image for image processing."
)

uploaded_image = st.file_uploader(
    "Upload ECG Image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
    key="ecg_image_upload"
)

if uploaded_image is not None:
    try:
        image = Image.open(
            BytesIO(uploaded_image.getvalue())
        ).convert("RGB")

        image_array = np.array(image)

        st.success("ECG Image Uploaded Successfully")

        st.image(
            image,
            caption="Original 12-Lead ECG",
            use_container_width=True
        )

        st.subheader("ECG Paper Calibration")

        col1, col2 = st.columns(2)

        with col1:
            paper_speed = st.selectbox(
                "Paper Speed (mm/s)",
                [5, 10, 12.5, 25, 50, 100],
                index=3
            )

        with col2:
            paper_gain = st.selectbox(
                "Amplitude Calibration (mm/mV)",
                [2.5, 5, 10, 20],
                index=2
            )

        pixels_per_mm = st.number_input(
            "Pixels per 1 mm grid square",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )

        seconds_per_pixel = (
            1.0 / paper_speed
        ) / pixels_per_mm

        millivolts_per_pixel = (
            1.0 / paper_gain
        ) / pixels_per_mm

        st.subheader("Calibration Information")

        st.write(
            f"Paper speed: {paper_speed} mm/s"
        )

        st.write(
            f"Amplitude calibration: {paper_gain} mm/mV"
        )

        st.write(
            f"Time per pixel: {seconds_per_pixel:.6f} seconds"
        )

        st.write(
            f"Voltage per pixel: {millivolts_per_pixel:.6f} mV"
        )

        hsv_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2HSV
        )

        lower_red_1 = np.array([0, 35, 50])
        upper_red_1 = np.array([15, 255, 255])

        lower_red_2 = np.array([160, 35, 50])
        upper_red_2 = np.array([180, 255, 255])

        red_mask_1 = cv2.inRange(
            hsv_image,
            lower_red_1,
            upper_red_1
        )

        red_mask_2 = cv2.inRange(
            hsv_image,
            lower_red_2,
            upper_red_2
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )

        cleaned_image = cv2.inpaint(
            image_array,
            red_mask,
            3,
            cv2.INPAINT_TELEA
        )

        gray_image = cv2.cvtColor(
            cleaned_image,
            cv2.COLOR_RGB2GRAY
        )

        enhanced_image = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(gray_image)

        binary_image = cv2.adaptiveThreshold(
            enhanced_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5
        )

        st.success("ECG Image Preprocessing Completed")

        st.write(
            "Image dimensions:",
            image_array.shape
        )

        st.write(
            "Image preprocessing: Completed"
        )

        st.write(
            "Color grid mask: Generated"
        )

        st.warning(
            "Image preprocessing is not equivalent to ECG "
            "signal extraction. Automatic 12-lead segmentation, "
            "waveform tracing, time alignment, voltage calibration "
            "and validation are still required before AI diagnosis."
        )

    except Exception as e:
        st.error(f"ECG Image Processing Error: {e}")


st.markdown("---")

st.markdown(
    "Project Developer: Amirabbas Sabervaskas  \n"
    "Academic Position: Third-Year Medical Student  \n"
    "University: Medical University of Sofia  \n"
    "Country: Bulgaria  \n"
    "Research Field: Artificial Intelligence in Cardiology  \n"
    "Project: ECG AI Diagnosis System  \n"
    "Year: 2026"
)

st.caption("Academic Research Project")
