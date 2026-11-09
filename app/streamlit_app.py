
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, MODELS_DIR, CLASS_NAMES
from src.features import extract_single


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
# Model loading
# --------------------------------------------------

AVAILABLE_MODELS = [
    "DenseNet121",
    "EfficientNetV2B0",
    "InceptionV3",
    "ResNet50",
    "VGG16"
]


@st.cache_resource
def load_svm(backbone):
    model_path = MODELS_DIR / f"{backbone}_svm.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


# --------------------------------------------------
# Image prediction
# --------------------------------------------------

st.header("🔍 Image Prediction")

st.write(
    "Upload an image and select a trained CNN + SVM model."
)

selected_model = st.selectbox(
    "Select Model",
    AVAILABLE_MODELS
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
                with st.spinner(
                    "Loading model and extracting image features..."
                ):

                    svm = load_svm(selected_model)

                    features = extract_single(
                        selected_model,
                        image
                    )

                    probabilities = svm.predict_proba(features)[0]

                    top_indices = probabilities.argsort()[::-1][:3]

                    predictions = [
                        {
                            "class": CLASS_NAMES[int(svm.classes_[i])],
                            "probability": float(probabilities[i])
                        }
                        for i in top_indices
                    ]

                st.success("Prediction completed!")

                best = predictions[0]

                st.metric(
                    "Predicted Class",
                    best["class"]
                )

                st.metric(
                    "Confidence",
                    f"{best['probability'] * 100:.2f}%"
                )

                st.subheader("Top 3 Predictions")

                for item in predictions:

                    st.write(
                        f"**{item['class']}** — "
                        f"{item['probability'] * 100:.2f}%"
                    )

                    st.progress(item["probability"])

            except Exception as e:
                st.error(f"Prediction error: {e}")


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

        st.subheader("Model Comparison")

        st.dataframe(
            s.round(4),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Performance Analysis")

        for img in ("comparison.png", "data_efficiency.png"):

            image_path = RESULTS_DIR / img

            if image_path.exists():
                st.image(
                    str(image_path),
                    use_container_width=True
                )

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

st.success("CNN + SVM Dashboard is ready.")
