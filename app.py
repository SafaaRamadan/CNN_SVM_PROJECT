
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import joblib
from PIL import Image



PROJECT_DIR = Path(__file__).resolve().parent

CNN_DIR = PROJECT_DIR / "saved_models" / "transfer_learning"
SVM_DIR = PROJECT_DIR / "svm_results"

MODEL_NAMES = ["ResNet50", "EfficientNetB0", "MobileNetV2"]

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PREPROCESS_FUNCTIONS = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
}



st.set_page_config(
    page_title="CNN + SVM | CIFAR-10",
    page_icon="🧠",
    layout="wide",
)



st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1120;
        color: #F1F5F9;
    }

    [data-testid="stMetric"] {
        background: #151F32;
        padding: 20px;
        border: 1px solid #263550;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.title("🧠 CNN + SVM Image Classification")

st.caption(
    "CIFAR-10 • Transfer Learning • Model Benchmarking"
)

st.divider()



col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Test Accuracy",
        "87.67%",
        "ResNet50",
    )

with col2:
    st.metric(
        "Fastest Training",
        "5.48 sec",
        "EfficientNetB0",
    )

with col3:
    st.metric(
        "Classes",
        "10",
        "CIFAR-10",
    )

st.divider()



st.subheader("📊 Model Performance Comparison")

results = pd.DataFrame(
    {
        "Model": [
            "ResNet50",
            "EfficientNetB0",
            "MobileNetV2",
        ],
        "Test Accuracy (%)": [
            87.67,
            87.0233,
            82.38,
        ],
        "Training Time (sec)": [
            14.45,
            5.48,
            5.83,
        ],
        "Accuracy Std (pp)": [
            0.04,
            0.0289,
            0.11,
        ],
    }
)

left, right = st.columns(2)

with left:
    st.markdown("**Test Accuracy**")

    st.bar_chart(
        results.set_index("Model")[["Test Accuracy (%)"]]
    )

with right:
    st.markdown("**SVM Training Time**")

    st.bar_chart(
        results.set_index("Model")[["Training Time (sec)"]]
    )

st.dataframe(
    results,
    hide_index=True,
    use_container_width=True,
)

st.divider()



@st.cache_resource
def load_models(model_name):
    """
    Load the selected CNN and SVM once and cache them.

    CNN:
        Used as a feature extractor.

    SVM:
        Predicts the class from the extracted features.
    """

    cnn_path = CNN_DIR / f"{model_name}.keras"
    svm_path = SVM_DIR / f"{model_name}_svm.joblib"

    if not cnn_path.is_file():
        raise FileNotFoundError(
            f"CNN model not found: {cnn_path}"
        )

    if not svm_path.is_file():
        raise FileNotFoundError(
            f"SVM model not found: {svm_path}"
        )

    # Load the saved CNN model.
    cnn = tf.keras.models.load_model(
        str(cnn_path),
        compile=False,
    )

    # Load the trained SVM.
    svm = joblib.load(svm_path)

    # Find the Global Average Pooling layer.
    pooling_layers = [
        layer
        for layer in cnn.layers
        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D,
        )
    ]

    if not pooling_layers:
        raise ValueError(
            f"No GlobalAveragePooling2D layer found "
            f"in {model_name}. Check the model architecture."
        )

    feature_layer = pooling_layers[-1]

    # Build a feature extractor from the trained CNN.
    extractor = tf.keras.Model(
        inputs=cnn.inputs,
        outputs=feature_layer.output,
        name=f"{model_name}_feature_extractor",
    )

    return extractor, svm



st.subheader("🖼️ Try Image Classification")

st.write(
    "Upload an image and select a model to classify it "
    "using CNN feature extraction and SVM."
)

model_name = st.selectbox(
    "Choose a model",
    MODEL_NAMES,
)

uploaded_file = st.file_uploader(
    "Upload a CIFAR-10 image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=250,
        )

    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        st.stop()

    if st.button("Classify Image", type="primary"):

        with st.spinner("Loading model and analyzing image..."):

            try:

                extractor, svm = load_models(model_name)

         
                # ResNet50, MobileNetV2 and EfficientNetB0
                # normally use 224x224 input images here.
                img_array = np.asarray(
                    image.resize((224, 224)),
                    dtype=np.float32,
                )

                # Shape: (224, 224, 3) -> (1, 224, 224, 3)
                img_array = np.expand_dims(
                    img_array,
                    axis=0,
                )

                # -----------------------------------------
                # Preprocess image
                # -----------------------------------------

                preprocess = PREPROCESS_FUNCTIONS[model_name]

                img_array = preprocess(img_array)

                # -----------------------------------------
                # Extract CNN features
                # -----------------------------------------

                features = extractor.predict(
                    img_array,
                    verbose=0,
                )

                # -----------------------------------------
                # SVM prediction
                # -----------------------------------------

                prediction = svm.predict(features)[0]

                # CIFAR-10 labels are normally integers 0-9.
                predicted_index = int(prediction)

                if not 0 <= predicted_index < len(CLASS_NAMES):
                    raise ValueError(
                        f"SVM returned unexpected class label: "
                        f"{prediction}"
                    )

                predicted_class = CLASS_NAMES[predicted_index]

                st.success(
                    f"Predicted Class: {predicted_class}"
                )

                st.write(f"**Selected model:** {model_name}")

                st.write(
                    f"**Extracted feature shape:** {features.shape}"
                )

                # -----------------------------------------
                # Optional confidence
                # -----------------------------------------

                if hasattr(svm, "predict_proba"):

                    probabilities = svm.predict_proba(features)[0]

                    confidence = float(
                        np.max(probabilities)
                    ) * 100

                    st.metric(
                        "SVM Probability for Most Likely Class",
                        f"{confidence:.2f}%",
                    )

                    st.caption(
                        "This is the SVM's estimated class probability, "
                        "not a guarantee that the prediction is correct."
                    )

            except Exception as e:

                st.error(f"Prediction failed: {e}")

                with st.expander("Technical details"):

                    st.write(
                        "Python executable:",
                        sys.executable,
                    )

                    st.write(
                        "TensorFlow version:",
                        tf.__version__,
                    )

                    st.write(
                        "Keras version:",
                        keras.__version__,
                    )

                    st.write(
                        "Selected model:",
                        model_name,
                    )

                    st.exception(e)




st.divider()

st.caption(
    "CNN feature extraction + SVM classification | CIFAR-10"
)
