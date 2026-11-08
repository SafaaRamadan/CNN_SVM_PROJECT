
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, CLASS_NAMES

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CNN + SVM | CIFAR-10",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Features + SVM Image Classifier")

st.caption(
    "CIFAR-10 • Frozen ImageNet backbones • "
    "RBF-SVM • Benchmarking and Image Prediction"
)

# --------------------------------------------------
# API configuration
# --------------------------------------------------

try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = "http://127.0.0.1:8000"

API_URL = API_URL.rstrip("/")
# --------------------------------------------------
# Image prediction
# --------------------------------------------------

st.header("🔍 Image Prediction")

st.write(
    "Upload a CIFAR-10 image and select a trained CNN + SVM model."
)

selected_model = st.selectbox(
    "Select Model",
    [
        "DenseNet121",
        "EfficientNetV2B0",
        "InceptionV3",
        "ResNet50",
        "VGG16"
    ]
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.write("**Selected model:**", selected_model)

        if st.button("Predict Image", type="primary"):

            try:
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    params={"model": selected_model},
                    timeout=120
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Prediction completed!")

                    st.metric(
                        "Predicted Class",
                        result.get(
                            "predicted_class",
                            result.get("class_name", "Unknown")
                        )
                    )

                    if "confidence" in result:
                        st.metric(
                            "Confidence",
                            f"{result['confidence'] * 100:.2f}%"
                        )

                    st.json(result)

                else:
                    st.error(
                        f"API Error {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the API server is running."
                )

            except requests.exceptions.Timeout:
                st.error("Prediction request timed out.")

            except Exception as e:
                st.error(f"Error: {e}")

# --------------------------------------------------
# Benchmark results
# --------------------------------------------------

st.divider()
st.header("📊 Benchmark Results")

summary_path = RESULTS_DIR / "summary.csv"

if not summary_path.exists():

    st.warning("Benchmark results not found: summary.csv")

else:

    s = pd.read_csv(summary_path)

    if s.empty:

        st.warning("The benchmark results file is empty.")

    else:

        best = s.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Best Test Accuracy",
            f"{best.test_acc_mean * 100:.2f}%",
            f"{best.backbone} + {best.classifier}"
        )

        c2.metric(
            "Backbones Tested",
            s.backbone.nunique()
        )

        c3.metric(
            "Classifiers Tested",
            s.classifier.nunique()
        )

        # Model comparison

        st.subheader("Model Comparison")

        st.dataframe(
            s.round(4),
            hide_index=True,
            use_container_width=True
        )

        # Performance analysis

        st.subheader("Performance Analysis")

        for img in ("comparison.png", "data_efficiency.png"):

            image_path = RESULTS_DIR / img

            if image_path.exists():
                st.image(
                    str(image_path),
                    use_container_width=True
                )

        # Confusion matrices

        st.subheader("Confusion Matrices")

        for image_path in sorted(
            RESULTS_DIR.glob("confusion_*.png")
        ):

            with st.expander(
                image_path.stem.replace("confusion_", "")
            ):

                st.image(
                    str(image_path),
                    use_container_width=True
                )

st.divider()

st.success("CNN + SVM Benchmark Dashboard is ready.")
