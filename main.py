from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# ============================
MODELS_DIR = Path("/content/drive/MyDrive/churn_prediction_business_pipeline/models")
LR_PATH = MODELS_DIR / "lr_churn_pipeline.joblib"
XGB_PATH = MODELS_DIR / "xgb_churn_pipeline.joblib"
THRESH_PATH = MODELS_DIR / "best_threshold.txt"
# ============================

app = FastAPI(title="Churn Prediction API")

# Load models (full sklearn Pipelines: preprocess + model
lr_model = joblib.load(LR_PATH))
xgb_model = joblib.load(XGB_PATH)

# Load optimized threshold
best_t = float(THRESH_PATH.read_text().strip())

class CustomerFeatures(BaseModel):
    # Send a JSON dict of features 
    features: dict

def _predict_proba(model, features: dict) -> float:
    # Convert dict -> single-row DataFrame
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[:, 1][0]  # probability of churn (label=1)
    return float(proba)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_dir": str(MODELS_DIR),
        "has_lr": LR_PATH.exists(),
        "has_xgb": XGB_PATH.exists(),
        "best_threshold": best_t,
    }

@app.post("/predict_lr")
def predict_lr(payload: CustomerFeatures):
    p = _predict_proba(lr_model, payload.features)
    return {"model": "logistic_regression", "churn_probability": p}

@app.post("/predict_xgb")
def predict_xgb(payload: CustomerFeatures):
    p = _predict_proba(xgb_model, payload.features)
    return {"model": "xgboost", "churn_probability": p}

@app.post("/predict_xgb_business")
def predict_xgb_business(payload: CustomerFeatures):
    p = _predict_proba(xgb_model, payload.features)
    action = "contact" if p >= best_t else "no_contact"
    return {
        "model": "xgboost",
        "churn_probability": p,
        "threshold": best_t,
        "decision": action
    }
