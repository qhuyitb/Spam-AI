import os
import joblib
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.bert_model import load_bert, predict_bert
from src.ml_models import load_model, predict as ml_predict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ML_MODELS = ("NaiveBayes", "SVM", "LogisticRegression")

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    for name in ML_MODELS:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            _models[name] = joblib.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  Skipped {name} (not found)")

    bert_model, bert_tokenizer = load_bert()
    _models["bert_model"] = bert_model
    _models["bert_tokenizer"] = bert_tokenizer
    print("  Loaded BERT from HuggingFace")
    print("All models ready.")
    yield
    _models.clear()


app = FastAPI(title="Spam Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ModelName = Literal["bert", "NaiveBayes", "SVM", "LogisticRegression"]


class PredictRequest(BaseModel):
    text: str
    model: ModelName = "bert"


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/models")
def list_models():
    available = []
    for name in ML_MODELS:
        available.append({"name": name, "type": "ml", "loaded": name in _models})
    available.append({"name": "bert", "type": "transformer", "loaded": "bert_model" in _models})
    return {"models": available}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text cannot be empty")

    if req.model == "bert":
        if "bert_model" not in _models:
            raise HTTPException(status_code=503, detail="BERT model not loaded")
        label, confidence = predict_bert(_models["bert_model"], _models["bert_tokenizer"], text)
    else:
        if req.model not in _models:
            raise HTTPException(status_code=503, detail=f"{req.model} not loaded")
        label, confidence = ml_predict(_models[req.model], text)

    return PredictResponse(text=text, label=label, confidence=round(confidence, 4), model=req.model)
