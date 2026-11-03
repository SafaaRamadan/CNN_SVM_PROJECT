import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, RESULTS_DIR, CLASS_NAMES, load_config
from src.features import extract_single

st.set_page_config(page_title="CNN + SVM | CIFAR-10", page_icon="🧠", layout="wide")
st.title("🧠 CNN Features + SVM Image Classifier")
st.caption("CIFAR-10 • Frozen ImageNet backbones • RBF-SVM • benchmarked against 4 baselines")

summary_path = RESULTS_DIR / "summary.csv"
if summary_path.exists():
    s = pd.read_csv(summary_path)
    best = s.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best test accuracy", f"{best.test_acc_mean*100:.2f}%", f"{best.backbone} + {best.classifier}")
    c2.metric("Backbones tested", s.backbone.nunique())
    c3.metric("Classifiers tested", s.classifier.nunique())
    st.dataframe(s.round(4), hide_index=True, use_container_width=True)
    for img in ("comparison.png", "data_efficiency.png"):
        if (RESULTS_DIR / img).exists():
            st.image(str(RESULTS_DIR / img))
else:
    st.warning("Run `python scripts/run_all.py` first to generate results.")

st.divider(); st.subheader("Try it")
available = sorted(p.name.replace("_svm.joblib", "") for p in MODELS_DIR.glob("*_svm.joblib"))
if not available:
    st.stop()
name = st.selectbox("Backbone", available)
if (RESULTS_DIR / f"confusion_{name}.png").exists():
    with st.expander("Confusion matrix on the CIFAR-10 test set"):
        st.image(str(RESULTS_DIR / f"confusion_{name}.png"))
threshold = st.slider("Low-confidence threshold", 0.0, 1.0, float(load_config()["low_confidence_threshold"]))
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB"); st.image(img, width=250)
    if st.button("Classify", type="primary"):
        svm = joblib.load(MODELS_DIR / f"{name}_svm.joblib")
        proba = svm.predict_proba(extract_single(name, img))[0]
        order = np.argsort(proba)[::-1]
        if proba[order[0]] < threshold:
            st.warning(f"Low confidence ({proba[order[0]]:.0%}): likely outside CIFAR-10's 10 classes or domain.")
        else:
            st.success(f"Prediction: {CLASS_NAMES[svm.classes_[order[0]]]} ({proba[order[0]]:.1%})")
        st.dataframe(pd.DataFrame({"Class": [CLASS_NAMES[svm.classes_[i]] for i in order[:3]],
                                   "Probability (%)": (proba[order[:3]] * 100).round(2)}), hide_index=True)
