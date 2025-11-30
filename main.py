import sys
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
from src.scripts.preprocessor import TextPreprocessor


sys.modules['__main__'].TextPreprocessor = TextPreprocessor
preprocessor = joblib.load("models/preprocessor.joblib")
classifier = joblib.load("models/classifier.joblib")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

app = FastAPI(title="Fake News Classifier API")

class NewsRequest(BaseModel):
    statement: str

class PredictionResponse(BaseModel):
    statement: str
    label: int
    probability: float

@app.get("/")
def welcome_page():
    return "Hi, Welcome"

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsRequest):
    raw_text = news.statement
    clean_text_str = clean_text(raw_text)

    X_final = preprocessor.transform([raw_text], [clean_text_str])

    prob = classifier.predict_proba(X_final)[:, 1][0]
    label = int(classifier.predict(X_final)[0])

    return PredictionResponse(statement=raw_text, label=label, probability=float(prob))