from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

# ── Load models once at startup ───────────────────────────────────────────────
with open('baseline_model.pkl', 'rb') as f:
    baseline_model = pickle.load(f)
with open('dro_model.pkl', 'rb') as f:
    dro_model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

app = FastAPI(
    title="DRO Credit Scoring API",
    description="Serves income predictions from a Distributionally Robust XGBoost model.",
    version="1.0.0"
)

# ── Request schema ────────────────────────────────────────────────────────────
class ApplicantProfile(BaseModel):
    AGEP: int = Field(..., ge=18, le=90, description="Age")
    COW: int = Field(..., ge=1, le=6, description="Class of worker")
    SCHL: int = Field(..., description="Education level code")
    MAR: int = Field(1, description="Marital status")
    OCCP: int = Field(..., description="Occupation code")
    POBP: int = Field(..., description="Place of birth code")
    RELP: int = Field(..., ge=0, le=17, description="Relationship to household")
    WKHP: int = Field(..., ge=1, le=99, description="Hours worked per week")
    SEX: int = Field(..., ge=1, le=2, description="Sex (1=Male, 2=Female)")
    RAC1P: int = Field(..., description="Race code")

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    model: str
    prediction: str
    prediction_label: int
    confidence: float

# ── Helper ────────────────────────────────────────────────────────────────────
def make_prediction(model, profile: ApplicantProfile):
    input_df = pd.DataFrame([profile.model_dump()])[feature_names]    label = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]
    confidence = float(proba[1] if label == 1 else proba[0])
    return {
        "prediction": "Above $50K" if label == 1 else "Below $50K",
        "prediction_label": label,
        "confidence": round(confidence, 4)
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "message": "DRO Credit Scoring API. See /docs for usage."}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict/baseline", response_model=PredictionResponse)
def predict_baseline(profile: ApplicantProfile):
    result = make_prediction(baseline_model, profile)
    return {"model": "baseline_erm", **result}

@app.post("/predict/dro", response_model=PredictionResponse)
def predict_dro(profile: ApplicantProfile):
    result = make_prediction(dro_model, profile)
    return {"model": "dro_robust", **result}