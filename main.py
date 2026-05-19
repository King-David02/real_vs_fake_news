import os
import sys
import re
import string
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import joblib
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Load custom preprocessor
from src.scripts.preprocessor import TextPreprocessor
sys.modules["__main__"].TextPreprocessor = TextPreprocessor

preprocessor = joblib.load("models/preprocessor.joblib")
classifier   = joblib.load("models/classifier.joblib")

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# App
app = FastAPI(title="Multilingual Fake News Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language support
LANGUAGE_NAMES = {
    "en": "English",
    "ig": "Igbo",
    "ha": "Hausa",
    "yo": "Yoruba",
}

def detect_language(text: str) -> tuple[str, str]:
    try:
        code = detect(text)
        if code not in LANGUAGE_NAMES:
            logger.info(f"Unsupported language ({code}), defaulting to English")
            return "en", "English"
        return code, LANGUAGE_NAMES[code]
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en", "English"

def translate_text(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    try:
        src = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt = LANGUAGE_NAMES.get(target_lang, target_lang)
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following text from {src} to {tgt}. "
                    "Only provide the translation, no explanations:\n\n" + text
                ),
            }],
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Schemas
class NewsRequest(BaseModel):
    statement: str
    search_query: Optional[str] = None
    force_language: Optional[str] = None

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: Optional[int] = None

class SourceReference(BaseModel):
    source_name: str
    url: str
    relevance: str

class PredictionResponse(BaseModel):
    statement: str
    original_statement: Optional[str] = None
    detected_language: Optional[str] = None
    language_code: Optional[str] = None
    was_translated: bool = False
    label: int
    label_text: str
    probability: float
    confidence: str
    search_results: List[SearchResult] = []
    analysis: str = ""
    mentioned_sources: List[SourceReference] = []

# Web search + analysis via OpenAI
def perform_web_search(
    statement: str, search_query: Optional[str] = None
) -> tuple[List[SearchResult], str, List[SourceReference]]:
    """
    Uses OpenAI's built-in web_search_preview tool to search the web
    and produce a fact-check analysis with cited sources.
    """
    query = search_query or f"fact check: {statement}"

    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=(
                f"Fact-check this news statement: \"{statement}\"\n\n"
                f"Search query to use: {query}\n\n"
                "Search the web and then provide:\n"
                "1. A numbered analysis of whether the statement is TRUE or FALSE based on what you find.\n"
                "2. For each point, cite the source using this exact format: [SOURCE: title | url]\n"
                "3. Be concise and factual."
            ),
        )

        analysis = ""
        search_results: List[SearchResult] = []
        mentioned_sources: List[SourceReference] = []
        seen_urls: set = set()

        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        analysis = block.text

                        # Extract citations from annotations (OpenAI populates these)
                        if hasattr(block, "annotations") and block.annotations:
                            for i, ann in enumerate(block.annotations, 1):
                                url   = getattr(ann, "url", None)
                                title = getattr(ann, "title", url or "Source")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    search_results.append(SearchResult(
                                        title=title, link=url, snippet="", position=i
                                    ))
                                    mentioned_sources.append(SourceReference(
                                        source_name=title, url=url, relevance=""
                                    ))

        # Fallback: parse [SOURCE: title | url] tags the model wrote inline
        if not mentioned_sources:
            for i, (title, url) in enumerate(
                re.findall(r"\[SOURCE:\s*([^|\]]+)\|\s*(https?://[^\]]+)\]", analysis), 1
            ):
                title, url = title.strip(), url.strip()
                if url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append(SearchResult(
                        title=title, link=url, snippet="", position=i
                    ))
                    mentioned_sources.append(SourceReference(
                        source_name=title, url=url, relevance=""
                    ))

        # Strip [SOURCE:...] tags from displayed analysis text
        analysis_clean = re.sub(r"\[SOURCE:[^\]]+\]", "", analysis).strip()

        return search_results, analysis_clean, mentioned_sources

    except Exception as e:
        logger.error(f"OpenAI web search error: {e}", exc_info=True)
        return [], f"Web search error: {str(e)}", []

# Routes
@app.get("/")
def root():
    return {"message": "Multilingual Fake News Classifier API", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsRequest):
    try:
        original_statement = news.statement

        if news.force_language and news.force_language in LANGUAGE_NAMES:
            lang_code = news.force_language
            lang_name = LANGUAGE_NAMES[lang_code]
        else:
            lang_code, lang_name = detect_language(original_statement)

        was_translated = lang_code != "en"
        english_text   = (
            translate_text(original_statement, "en", lang_code)
            if was_translated else original_statement
        )

        X      = preprocessor.transform([english_text], [clean_text(english_text)])
        prob   = classifier.predict_proba(X)[:, 1][0]
        label  = int(classifier.predict(X)[0])

        label_text = "FAKE" if label == 0 else "REAL"
        confidence = (
            "High"   if prob > 0.75 or prob < 0.25 else
            "Medium" if prob > 0.60 or prob < 0.40 else
            "Low"
        )

        search_results, analysis, mentioned_sources = perform_web_search(
            english_text, news.search_query
        )

        if was_translated and analysis:
            analysis = translate_text(analysis, lang_code, "en")

        return PredictionResponse(
            statement          = original_statement,
            original_statement = original_statement if was_translated else None,
            detected_language  = lang_name,
            language_code      = lang_code,
            was_translated     = was_translated,
            label              = label,
            label_text         = label_text,
            probability        = float(prob),
            confidence         = confidence,
            search_results     = search_results,
            analysis           = analysis,
            mentioned_sources  = mentioned_sources,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)