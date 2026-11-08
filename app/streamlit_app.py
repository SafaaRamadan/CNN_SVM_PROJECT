
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR

# Page configuration
st.set_page_config(
    page_title="CNN + SVM | CIFAR-10",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Features + SVM Image Classifier")

st.caption(
    "CIFAR-10 • Frozen ImageNet backbones • "
    "RBF-SVM • Benchmarked against 4 baselines"
)

st.header("Benchmark Results")

summary_path = RESULTS_DIR / "summary.csv"

if not summary_path.exists():
    st.warning(
        "Benchmark results not found. "
        "Please generate summary.csv first."
    )
    st.stop()

s = pd.read_csv(summary_path)

if s.empty:
    st.warning("The benchmark results file is empty.")
    st.stop()

# Best model
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

# Full comparison table
st.subheader("Model Comparison")

st.dataframe(
    s.round(4),
    hide_index=True,
    use_container_width=True
)

# Comparison charts
st.subheader("Performance Analysis")

for img in ("comparison.png", "data_efficiency.png"):

    image_path = RESULTS_DIR / img

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(f"{img} is not available.")

# Confusion matrices
st.subheader("Confusion Matrices")

for image_path in sorted(RESULTS_DIR.glob("confusion_*.png")):

    with st.expander(image_path.stem.replace("confusion_", "")):
        st.image(str(image_path), use_container_width=True)

st.divider()

st.success(
    "Benchmark dashboard loaded successfully. "
    "Image prediction through FastAPI is temporarily disabled."
)
