import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import string
from typing import List, Optional
from mistralai import Mistral
from atp_sdk.clients import LLMClient
import logging
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Load TextPreprocessor
from src.scripts.preprocessor import TextPreprocessor
sys.modules['__main__'].TextPreprocessor = TextPreprocessor

# Load models
preprocessor = joblib.load("models/preprocessor.joblib")
classifier = joblib.load("models/classifier.joblib")

# Initialize Mistral and ATP clients
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ATP_API_KEY = os.getenv("ATP_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
llm_client = LLMClient(api_key=ATP_API_KEY, protocol="https")

app = FastAPI(title="Multilingual Fake News Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RESTRICTED LANGUAGE MAP (ONLY CHANGE)
LANGUAGE_NAMES = {
    "en": "English",
    "ig": "Igbo",
    "ha": "Hausa",
    "yo": "Yoruba",
}

def detect_language(text: str) -> tuple[str, str]:
    try:
        lang_code = detect(text)

        if lang_code not in LANGUAGE_NAMES:
            logger.info(f"Unsupported language ({lang_code}), defaulting to English")
            return "en", "English"

        return lang_code, LANGUAGE_NAMES[lang_code]

    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return "en", "English"

def translate_text(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    try:
        prompt = (
            f"Translate the following text from {LANGUAGE_NAMES.get(source_lang, source_lang)} "
            f"to {LANGUAGE_NAMES.get(target_lang, target_lang)}. "
            "Only provide the translation, no explanations:\n\n"
            f"{text}"
        )

        response = mistral_client.chat.complete(
            model="mistral-large-2411",
            messages=[{"role": "user", "content": prompt}],
        )

        return extract_text_from_response(response.choices[0].message.content).strip()

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_response(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join(
            chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in content
        )
    return str(content)

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

TOOL_PLANNER_PROMPT = {
    "role": "system",
    "content": (
        "You are a search tool planner. "
        "Use the available search tools to find relevant information about the news statement. "
        "Focus on finding credible sources that can verify or refute the claim."
    ),
}

FINAL_RESPONDER_PROMPT = {
    "role": "system",
    "content": (
        "You are a fact-checking assistant. "
        "Based on the search results provided, analyze whether the news statement is likely true or false. "
        "When you mention a source in your analysis, use this exact format: [SOURCE: exact title from search results]. "
        "Provide a brief, clear analysis in numbered points."
    ),
}

def perform_web_search(statement: str, search_query: Optional[str] = None):
    try:
        toolkit_id = "serper_toolkit"
        query = search_query or f"fact check: {statement}"

        context = llm_client.get_toolkit_context(
            toolkit_id=toolkit_id,
            provider="mistralai",
            user_prompt=query,
        )

        plan = mistral_client.chat.complete(
            model="mistral-large-2411",
            messages=[TOOL_PLANNER_PROMPT, {"role": "user", "content": query}],
            tools=context["tools"],
            tool_choice="auto",
        )

        tool_calls = plan.choices[0].message.tool_calls
        if not tool_calls:
            return [], "No search results available.", []

        results = llm_client.call_tool(
            toolkit_id=toolkit_id,
            tool_calls=tool_calls,
            provider="mistralai",
            user_prompt=query,
            auth_token=SERPER_API_KEY,
        )

        search_results = []
        for r in results:
            import json
            content = json.loads(r["content"])
            organic = content.get("organic") or content.get("organic_results") or []
            for i, item in enumerate(organic[:10], 1):
                search_results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        position=i,
                    )
                )

        final = mistral_client.chat.complete(
            model="mistral-large-2411",
            messages=[FINAL_RESPONDER_PROMPT, {"role": "user", "content": query}],
        )

        analysis = extract_text_from_response(final.choices[0].message.content)
        return search_results, analysis, []

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return [], f"Search error: {str(e)}", []

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

        english_text = (
            translate_text(original_statement, "en", lang_code)
            if was_translated
            else original_statement
        )

        X = preprocessor.transform([english_text], [clean_text(english_text)])
        prob = classifier.predict_proba(X)[:, 1][0]
        label = int(classifier.predict(X)[0])

        label_text = "FAKE" if label == 0 else "REAL"
        confidence = (
            "High" if prob > 0.75 or prob < 0.25
            else "Medium" if prob > 0.6 or prob < 0.4
            else "Low"
        )

        search_results, analysis, mentioned_sources = perform_web_search(english_text)

        if was_translated and analysis:
            analysis = translate_text(analysis, lang_code, "en")

        return PredictionResponse(
            statement=original_statement,
            original_statement=original_statement if was_translated else None,
            detected_language=lang_name,
            language_code=lang_code,
            was_translated=was_translated,
            label=label,
            label_text=label_text,
            probability=float(prob),
            confidence=confidence,
            search_results=search_results,
            analysis=analysis,
            mentioned_sources=mentioned_sources,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
