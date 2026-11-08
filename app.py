import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    "Upload a 12-lead ECG signal to analyze it "
    "using an Artificial Intelligence model."
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


uploaded_file = st.file_uploader(
    "Upload ECG File (.npy)",
    type=["npy"]
)


if uploaded_file is not None:

    try:
        ecg = np.load(
            uploaded_file,
            allow_pickle=False
        )

        if ecg.shape != (1000, 12):
            st.error(
                f"Incorrect ECG shape: {ecg.shape}. "
                "Expected (1000, 12)."
            )
            st.stop()

        if not np.issubdtype(ecg.dtype, np.number):
            st.error("ECG data must be numeric.")
            st.stop()

        if not np.all(np.isfinite(ecg)):
            st.error("ECG contains invalid values.")
            st.stop()

        ecg = ecg.astype(np.float32)

        st.success("ECG Uploaded Successfully")

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
            st.stop()

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
                "AI Score (%)": round(
                    percentage,
                    2
                )
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
            "AI predictions are not a substitute "
            "for professional medical diagnosis."
        )

    except Exception as e:

        st.error(
            f"ECG Analysis Error: {e}"
        )
