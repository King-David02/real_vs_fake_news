import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import string
from typing import List, Dict, Optional
from mistralai import Mistral
from atp_sdk.clients import LLMClient
import logging
from dotenv import load_dotenv

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

app = FastAPI(title="Fake News Classifier with Web Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_response(content):
    """Extract text from Mistral API response (handles both string and TextChunk list)"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If content is a list of TextChunk objects
        return "".join([
            chunk.text if hasattr(chunk, 'text') else str(chunk) 
            for chunk in content
        ])
    else:
        return str(content)

class NewsRequest(BaseModel):
    statement: str
    search_query: Optional[str] = None

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
    )
}

FINAL_RESPONDER_PROMPT = {
    "role": "system",
    "content": (
        "You are a fact-checking assistant. "
        "Based on the search results provided, analyze whether the news statement is likely true or false. "
        "When you mention a source in your analysis, use this exact format: [SOURCE: exact title from search results]. "
        "For example: 'According to [SOURCE: Premium Times Nigeria] the event is confirmed.' "
        "Provide a brief, clear analysis in numbered points. "
        "Be objective and point out if information is inconclusive. "
        "Always reference sources using the [SOURCE: ...] format when making claims."
    )
}

def perform_web_search(statement: str, search_query: Optional[str] = None):
    """Perform web search using Mistral and ATP SDK"""
    try:
        toolkit_id = "serper_toolkit"
        auth_token = SERPER_API_KEY
        provider = "mistralai"
        
        # Use custom search query or create one from statement
        query = search_query or f"fact check: {statement}"
        
        # Get toolkit context
        context = llm_client.get_toolkit_context(
            toolkit_id=toolkit_id,
            provider=provider,
            user_prompt=query
        )
        
        conversation_history = [{"role": "user", "content": query}]
        
        # Tool Planner Step
        plan_response = mistral_client.chat.complete(
            model="mistral-large-2411",
            messages=[TOOL_PLANNER_PROMPT] + conversation_history,
            tools=context["tools"],
            tool_choice="auto"
        )
        
        tool_calls = plan_response.choices[0].message.tool_calls
        
        if not tool_calls:
            return [], "No search results available."
        
        # Execute tool calls
        results = llm_client.call_tool(
            toolkit_id=toolkit_id,
            tool_calls=tool_calls,
            provider=provider,
            user_prompt=query,
            auth_token=auth_token
        )
        
        # Append assistant message with tool calls
        conversation_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        })
        
        # Append tool results
        for result in results:
            conversation_history.append({
                "role": result["role"],
                "name": result["name"],
                "content": result["content"],
                "tool_call_id": result["tool_call_id"]
            })
        
        # Final Responder Step
        final_response = mistral_client.chat.complete(
            model="mistral-large-2411",
            messages=[FINAL_RESPONDER_PROMPT] + conversation_history
        )
        
        # Extract text from response
        analysis = extract_text_from_response(final_response.choices[0].message.content)
        
        # Clean up any reference citations from the analysis
        import re
        analysis = re.sub(r'reference_ids=\[[^\]]*\]\s*type=[\'"]reference[\'"]', '', analysis)
        analysis = re.sub(r'\s+', ' ', analysis).strip()
        
        # Parse search results
        search_results = []
        for result in results:
            import json
            try:
                content = json.loads(result["content"])
                if "organic" in content:
                    for idx, item in enumerate(content["organic"][:10], 1):  # Get top 10
                        search_results.append(SearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            position=idx
                        ))
            except Exception as parse_error:
                logger.warning(f"Error parsing search result: {parse_error}")
                continue
        
        # Extract mentioned sources from analysis and match with URLs
        mentioned_sources = []
        if search_results and analysis:
            # Find all [SOURCE: ...] mentions in the analysis
            import re
            source_mentions = re.findall(r'\[SOURCE:\s*([^\]]+)\]', analysis)
            
            for mention in source_mentions:
                mention_lower = mention.lower().strip()
                # Try to match with actual search results
                for result in search_results:
                    title_lower = result.title.lower()
                    # Check if the mention matches the title (fuzzy match)
                    if mention_lower in title_lower or any(word in title_lower for word in mention_lower.split() if len(word) > 3):
                        if not any(m.url == result.link for m in mentioned_sources):
                            mentioned_sources.append(SourceReference(
                                source_name=result.title,
                                url=result.link,
                                relevance=f"Mentioned as: {mention}"
                            ))
                        break
            
            # Clean up the [SOURCE: ...] tags from analysis for display
            analysis = re.sub(r'\[SOURCE:\s*([^\]]+)\]', r'"\1"', analysis)
        
        return search_results, analysis, mentioned_sources
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return [], f"Search error: {str(e)}"

@app.get("/")
def welcome_page():
    return {"message": "Fake News Classifier API with Web Search", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsRequest):
    try:
        raw_text = news.statement
        clean_text_str = clean_text(raw_text)
        
        # Make prediction
        X_final = preprocessor.transform([raw_text], [clean_text_str])
        prob = classifier.predict_proba(X_final)[:, 1][0]
        label = int(classifier.predict(X_final)[0])
        
        # Determine label text and confidence
        label_text = "FAKE" if label == 0 else "REAL"
        confidence = "High" if prob > 0.75 or prob < 0.25 else "Medium" if prob > 0.6 or prob < 0.4 else "Low"
        
        # Perform web search
        search_results, analysis, mentioned_sources = perform_web_search(raw_text, news.search_query)
        
        return PredictionResponse(
            statement=raw_text,
            label=label,
            label_text=label_text,
            probability=float(prob),
            confidence=confidence,
            search_results=search_results,
            analysis=analysis,
            mentioned_sources=mentioned_sources
        )
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-only")
def search_only(news: NewsRequest):
    """Endpoint for web search only without classification"""
    try:
        search_results, analysis, mentioned_sources = perform_web_search(news.statement, news.search_query)
        return {
            "statement": news.statement,
            "search_results": search_results,
            "analysis": analysis,
            "mentioned_sources": mentioned_sources
        }
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)