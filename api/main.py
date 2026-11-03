"""uvicorn api.main:app --port 8000   ->   POST /predict (multipart image), GET /health"""
import io, sys
from pathlib import Path
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, CLASS_NAMES
from src.features import extract_single

app = FastAPI(title="CNN+SVM CIFAR-10 API")
_models = {}


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(p.name.replace("_svm.joblib", "") for p in MODELS_DIR.glob("*_svm.joblib"))}


@app.post("/predict")
async def predict(file: UploadFile = File(...), backbone: str = "DenseNet121"):
    path = MODELS_DIR / f"{backbone}_svm.joblib"
    if not path.exists():
        raise HTTPException(404, f"No model for {backbone}")
    if backbone not in _models:
        _models[backbone] = joblib.load(path)
    svm = _models[backbone]
    img = Image.open(io.BytesIO(await file.read()))
    proba = svm.predict_proba(extract_single(backbone, img))[0]
    top = proba.argsort()[::-1][:3]
    return {"backbone": backbone,
            "predictions": [{"class": CLASS_NAMES[svm.classes_[i]], "probability": float(proba[i])} for i in top]}
