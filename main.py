# import sys
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import re
# import string
# from src.scripts.preprocessor import TextPreprocessor


# sys.modules['__main__'].TextPreprocessor = TextPreprocessor
# preprocessor = joblib.load("models/preprocessor.joblib")
# classifier = joblib.load("models/classifier.joblib")

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(f"[{string.punctuation}]", "", text)
#     text = re.sub(r"\d+", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# app = FastAPI(title="Fake News Classifier API")

# class NewsRequest(BaseModel):
#     statement: str

# class PredictionResponse(BaseModel):
#     statement: str
#     label: int
#     probability: float

# @app.get("/")
# def welcome_page():
#     return "Hi, Welcome"

# @app.post("/predict", response_model=PredictionResponse)
# def predict(news: NewsRequest):
#     raw_text = news.statement
#     clean_text_str = clean_text(raw_text)

#     X_final = preprocessor.transform([raw_text], [clean_text_str])

#     prob = classifier.predict_proba(X_final)[:, 1][0]
#     label = int(classifier.predict(X_final)[0])

#     return PredictionResponse(statement=raw_text, label=label, probability=float(prob))


import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import string
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import json
import requests

# Load environment variables
load_dotenv()

from src.scripts.preprocessor import TextPreprocessor
sys.modules['__main__'].TextPreprocessor = TextPreprocessor

# Load models
preprocessor = joblib.load("models/preprocessor.joblib")
classifier = joblib.load("models/classifier.joblib")

# Initialize Bing News Search
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY")
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"

# Initialize Groq API (Free and Fast)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

# Determine which LLM to use
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Trusted news sources with credibility scores
TRUSTED_DOMAINS = {
    'reuters.com': 0.95,
    'apnews.com': 0.95,
    'bbc.com': 0.90,
    'bbc.co.uk': 0.90,
    'nytimes.com': 0.85,
    'washingtonpost.com': 0.85,
    'theguardian.com': 0.85,
    'wsj.com': 0.85,
    'cnn.com': 0.80,
    'npr.org': 0.85,
    'bloomberg.com': 0.85,
    'usatoday.com': 0.75,
    'forbes.com': 0.75,
    'time.com': 0.80,
    'newsweek.com': 0.75,
    'politico.com': 0.80,
    'thehill.com': 0.75,
    'cnbc.com': 0.80,
    'abcnews.go.com': 0.80,
    'cbsnews.com': 0.80,
    'nbcnews.com': 0.80,
    'foxnews.com': 0.70,
    'msnbc.com': 0.75,
    'axios.com': 0.80,
    'economist.com': 0.85,
    'ft.com': 0.85,
    'aljazeera.com': 0.80,
    'dw.com': 0.80,
}

SUSPICIOUS_DOMAINS = {
    'naturalnews.com': 0.2,
    'infowars.com': 0.2,
    'beforeitsnews.com': 0.2,
}


def clean_text(text):
    """Clean text for model prediction"""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_claims_with_groq(text: str, max_claims: int = 3) -> List[str]:
    """Use Groq API to extract searchable claims"""
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found, falling back to simple extraction")
        return extract_claims_simple(text, max_claims)
    
    try:
        prompt = f"""You are a fact-checking assistant. Extract {max_claims} specific, verifiable factual claims from the following text that can be searched on news websites.

Rules:
1. Each claim should be a clear, searchable phrase (5-15 words)
2. Focus on specific facts, events, or statements that can be verified
3. Remove opinions and focus on factual assertions
4. Make claims specific enough to search but general enough to find sources
5. Return ONLY a JSON array of strings, nothing else

Text to analyze:
"{text}"

Return format: ["claim 1", "claim 2", "claim 3"]"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            content = content.replace('```json', '').replace('```', '').strip()
            claims = json.loads(content)
            
            if isinstance(claims, list) and claims:
                return claims[:max_claims]
            else:
                return extract_claims_simple(text, max_claims)
        else:
            return extract_claims_simple(text, max_claims)
            
    except Exception as e:
        print(f"Error extracting claims with Groq: {str(e)}")
        return extract_claims_simple(text, max_claims)


def extract_claims_with_ollama(text: str, max_claims: int = 3) -> List[str]:
    """Use Ollama (local) to extract searchable claims"""
    try:
        prompt = f"""You are a fact-checking assistant. Extract {max_claims} specific, verifiable factual claims from the following text that can be searched on news websites.

Rules:
1. Each claim should be a clear, searchable phrase (5-15 words)
2. Focus on specific facts, events, or statements that can be verified
3. Remove opinions and focus on factual assertions
4. Make claims specific enough to search but general enough to find sources
5. Return ONLY a JSON array of strings, nothing else

Text to analyze:
"{text}"

Return format: ["claim 1", "claim 2", "claim 3"]"""

        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['response'].strip()
            content = content.replace('```json', '').replace('```', '').strip()
            claims = json.loads(content)
            
            if isinstance(claims, list) and claims:
                return claims[:max_claims]
            else:
                return extract_claims_simple(text, max_claims)
        else:
            return extract_claims_simple(text, max_claims)
            
    except Exception as e:
        print(f"Error extracting claims with Ollama: {str(e)}")
        return extract_claims_simple(text, max_claims)


def extract_claims_simple(text: str, max_claims: int = 3) -> List[str]:
    """Fallback: Simple claim extraction without LLM"""
    sentences = re.split(r'[.!?]+', text)
    claims = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        word_count = len(sentence.split())
        if 5 <= word_count <= 15:
            claims.append(sentence)
    
    if not claims:
        words = text.split()[:10]
        claims = [' '.join(words)]
    
    return claims[:max_claims]


def extract_key_claims(text: str, max_claims: int = 3) -> List[str]:
    """Main function to extract claims based on configured LLM provider"""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return extract_claims_with_groq(text, max_claims)
    elif LLM_PROVIDER == "ollama":
        return extract_claims_with_ollama(text, max_claims)
    else:
        return extract_claims_simple(text, max_claims)


def calculate_source_credibility(url: str) -> float:
    """Calculate credibility score for a source based on domain"""
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        
        if domain in TRUSTED_DOMAINS:
            return TRUSTED_DOMAINS[domain]
        elif domain in SUSPICIOUS_DOMAINS:
            return SUSPICIOUS_DOMAINS[domain]
        else:
            return 0.5
    except:
        return 0.5


def search_news_with_bing(claims: List[str], max_results: int = 5) -> List[Dict]:
    """
    Search for news articles using Bing News Search API
    Bing provides diverse sources, not limited to specific sites
    """
    if not BING_SEARCH_KEY:
        print("BING_SEARCH_KEY not found")
        return []
    
    all_sources = []
    
    for claim in claims:
        try:
            # Bing News Search API request
            headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}
            params = {
                "q": claim,
                "count": max_results,
                "mkt": "en-US",  # Market
                "freshness": "Month",  # Articles from past month
                "sortBy": "Relevance",  # Sort by relevance
                "textDecorations": False,
                "textFormat": "Raw"
            }
            
            response = requests.get(
                BING_SEARCH_ENDPOINT,
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('value', [])
                
                for article in articles:
                    url = article.get('url', '')
                    
                    # Extract source name from provider or URL
                    provider = article.get('provider', [{}])[0] if article.get('provider') else {}
                    source_name = provider.get('name', urlparse(url).netloc if url else 'Unknown')
                    
                    # Parse date
                    date_published = article.get('datePublished', '')
                    if date_published:
                        try:
                            # Bing format: 2024-01-15T10:30:00.0000000Z
                            date_obj = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                            date_formatted = date_obj.strftime('%Y-%m-%d')
                        except:
                            date_formatted = date_published[:10] if len(date_published) >= 10 else ''
                    else:
                        date_formatted = ''
                    
                    source_data = {
                        'title': article.get('name', 'No title'),
                        'url': url,
                        'source': source_name,
                        'description': article.get('description', ''),
                        'published_at': date_formatted,
                        'credibility': calculate_source_credibility(url),
                        'claim_searched': claim,
                        # Additional Bing-specific metadata
                        'image_url': article.get('image', {}).get('thumbnail', {}).get('contentUrl', '') if article.get('image') else '',
                        'category': article.get('category', '')
                    }
                    all_sources.append(source_data)
            
            elif response.status_code == 401:
                print("Bing API: Invalid subscription key")
                return []
            elif response.status_code == 403:
                print("Bing API: Rate limit exceeded or quota used up")
                return []
            else:
                print(f"Bing API error: {response.status_code}")
        
        except Exception as e:
            print(f"Error searching for claim '{claim}': {str(e)}")
            continue
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_sources = []
    for source in all_sources:
        if source['url'] not in seen_urls:
            seen_urls.add(source['url'])
            unique_sources.append(source)
    
    # Sort by credibility score
    unique_sources.sort(key=lambda x: x['credibility'], reverse=True)
    
    return unique_sources[:10]


def generate_final_conclusion(
    model_label: int,
    model_probability: float,
    sources: List[Dict]
) -> Dict:
    """Generate a final conclusion combining model prediction and web sources"""
    conclusion = {
        'verdict': '',
        'confidence': '',
        'explanation': '',
        'recommendation': ''
    }
    
    if not sources:
        if model_label == 1:
            conclusion['verdict'] = 'LIKELY REAL'
            conclusion['confidence'] = 'MODERATE'
            conclusion['explanation'] = (
                f"The model predicts this is real news with {model_probability*100:.1f}% confidence. "
                "However, no corroborating sources were found online. "
                "This could mean the news is very recent or not widely reported yet."
            )
        else:
            conclusion['verdict'] = 'LIKELY FAKE'
            conclusion['confidence'] = 'MODERATE'
            conclusion['explanation'] = (
                f"The model predicts this is fake news with {(1-model_probability)*100:.1f}% confidence. "
                "No credible sources were found to verify this claim online."
            )
        conclusion['recommendation'] = (
            "Always verify important information from multiple trusted news sources."
        )
        return conclusion
    
    # Calculate average credibility of sources
    avg_credibility = sum(s['credibility'] for s in sources) / len(sources)
    high_credibility_count = sum(1 for s in sources if s['credibility'] >= 0.80)
    
    # Decision logic combining model and sources
    if model_label == 1 and avg_credibility >= 0.75:
        conclusion['verdict'] = 'VERIFIED REAL'
        conclusion['confidence'] = 'HIGH'
        conclusion['explanation'] = (
            f"The model predicts this is real news with {model_probability*100:.1f}% confidence, "
            f"and this is corroborated by {high_credibility_count} high-credibility sources. "
            f"The average credibility score of sources is {avg_credibility:.2f}."
        )
        conclusion['recommendation'] = (
            "This claim appears to be legitimate based on both AI analysis and credible news sources."
        )
    
    elif model_label == 1 and avg_credibility < 0.75:
        conclusion['verdict'] = 'NEEDS VERIFICATION'
        conclusion['confidence'] = 'LOW'
        conclusion['explanation'] = (
            f"The model predicts this is real news with {model_probability*100:.1f}% confidence, "
            f"but the sources found have lower credibility (average: {avg_credibility:.2f}). "
            "This discrepancy suggests caution is needed."
        )
        conclusion['recommendation'] = (
            "Cross-check this information with more authoritative news sources before trusting it."
        )
    
    elif model_label == 0 and avg_credibility >= 0.75:
        conclusion['verdict'] = 'CONFLICTING SIGNALS'
        conclusion['confidence'] = 'UNCERTAIN'
        conclusion['explanation'] = (
            f"The model predicts this might be fake news with {(1-model_probability)*100:.1f}% confidence, "
            f"but {high_credibility_count} credible sources report on this topic. "
            "The claim may be partially true but exaggerated or misrepresented."
        )
        conclusion['recommendation'] = (
            "Review the actual sources to understand the full context. "
            "The claim might contain some truth but be presented misleadingly."
        )
    
    else:
        conclusion['verdict'] = 'LIKELY FAKE'
        conclusion['confidence'] = 'HIGH'
        conclusion['explanation'] = (
            f"The model predicts this is fake news with {(1-model_probability)*100:.1f}% confidence, "
            f"and the sources found have low credibility (average: {avg_credibility:.2f}). "
            "This strongly suggests the claim is unreliable."
        )
        conclusion['recommendation'] = (
            "This claim is very likely false. Do not share without verification from trusted sources."
        )
    
    return conclusion


app = FastAPI(title="Fake News Classifier API with Bing News Search")


class NewsRequest(BaseModel):
    statement: str


class SourceInfo(BaseModel):
    title: str
    url: str
    source: str
    description: str
    published_at: str
    credibility: float
    claim_searched: str


class ConclusionInfo(BaseModel):
    verdict: str
    confidence: str
    explanation: str
    recommendation: str


class PredictionResponse(BaseModel):
    statement: str
    label: int
    probability: float
    claims_extracted: List[str]
    sources_found: List[SourceInfo]
    sources_count: int
    final_conclusion: ConclusionInfo
    search_enabled: bool
    llm_provider: str


@app.get("/")
def welcome_page():
    return {
        "message": "Fake News Classifier API with Bing News Search",
        "version": "2.5",
        "search_provider": "Bing News Search",
        "search_enabled": BING_SEARCH_KEY is not None,
        "llm_provider": LLM_PROVIDER,
        "llm_available": (LLM_PROVIDER == "groq" and GROQ_API_KEY is not None) or 
                        (LLM_PROVIDER == "ollama")
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsRequest):
    """
    Predict if news is fake or real, use LLM to extract claims,
    search Bing News for sources and provide conclusion
    """
    try:
        raw_text = news.statement
        
        # Step 1: Get model prediction
        clean_text_str = clean_text(raw_text)
        X_final = preprocessor.transform([raw_text], [clean_text_str])
        prob = classifier.predict_proba(X_final)[:, 1][0]
        label = int(classifier.predict(X_final)[0])
        
        # Step 2: Extract key claims using LLM
        claims = extract_key_claims(raw_text, max_claims=2)
        
        # Step 3: Search for sources using Bing News
        sources = []
        if BING_SEARCH_KEY and claims:
            sources = search_news_with_bing(claims, max_results=5)
        
        # Step 4: Generate final conclusion
        conclusion = generate_final_conclusion(label, prob, sources)
        
        return PredictionResponse(
            statement=raw_text,
            label=label,
            probability=float(prob),
            claims_extracted=claims,
            sources_found=sources,
            sources_count=len(sources),
            final_conclusion=conclusion,
            search_enabled=BING_SEARCH_KEY is not None,
            llm_provider=LLM_PROVIDER
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/health")
def health_check():
    llm_status = "none"
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        llm_status = "groq (ready)"
    elif LLM_PROVIDER == "ollama":
        try:
            response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=2)
            llm_status = "ollama (ready)" if response.status_code == 200 else "ollama (not running)"
        except:
            llm_status = "ollama (not running)"
    
    # Check Bing API status
    bing_status = "not configured"
    if BING_SEARCH_KEY:
        try:
            # Quick test request
            headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY}
            test_response = requests.get(
                BING_SEARCH_ENDPOINT,
                headers=headers,
                params={"q": "test", "count": 1},
                timeout=5
            )
            if test_response.status_code == 200:
                bing_status = "ready"
            elif test_response.status_code == 401:
                bing_status = "invalid key"
            elif test_response.status_code == 403:
                bing_status = "quota exceeded"
            else:
                bing_status = f"error ({test_response.status_code})"
        except:
            bing_status = "connection error"
    
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "preprocessor_loaded": preprocessor is not None,
        "search_provider": "Bing News Search",
        "search_enabled": BING_SEARCH_KEY is not None,
        "bing_status": bing_status,
        "llm_provider": LLM_PROVIDER,
        "llm_status": llm_status
    }