import os
import sys
import re
import json
import string
import logging
from datetime import datetime
from typing import List, Optional

import joblib
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

load_dotenv()
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load preprocessor & models
from src.scripts.preprocessor import TextPreprocessor
sys.modules["__main__"].TextPreprocessor = TextPreprocessor

preprocessor = joblib.load("models/preprocessor.joblib")
classifier   = joblib.load("models/classifier.joblib")

# ── OpenAI client — works locally (.env) and on Streamlit Cloud (st.secrets) ─
def get_secret(key: str) -> Optional[str]:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

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
            return "en", "English"
        return code, LANGUAGE_NAMES[code]
    except Exception:
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

# ML prediction
def run_model(english_text: str) -> tuple[int, str, float, str]:
    X      = preprocessor.transform([english_text], [clean_text(english_text)])
    prob   = float(classifier.predict_proba(X)[:, 1][0])
    label  = int(classifier.predict(X)[0])
    label_text = "FAKE" if label == 0 else "REAL"
    confidence = (
        "High"   if prob > 0.75 or prob < 0.25 else
        "Medium" if prob > 0.60 or prob < 0.40 else
        "Low"
    )
    return label, label_text, prob, confidence

# Web search + analysis
def perform_web_search(
    statement: str, search_query: Optional[str] = None
) -> tuple[list, str, list]:
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
        search_results = []
        mentioned_sources = []
        seen_urls: set = set()

        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        analysis = block.text

                        # Primary: extract from OpenAI annotations
                        if hasattr(block, "annotations") and block.annotations:
                            for i, ann in enumerate(block.annotations, 1):
                                url   = getattr(ann, "url", None)
                                title = getattr(ann, "title", url or "Source")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    search_results.append({"title": title, "link": url, "snippet": "", "position": i})
                                    mentioned_sources.append({"source_name": title, "url": url, "relevance": ""})

        # Fallback: parse [SOURCE: title | url] written inline by the model
        if not mentioned_sources:
            for i, (title, url) in enumerate(
                re.findall(r"\[SOURCE:\s*([^|\]]+)\|\s*(https?://[^\]]+)\]", analysis), 1
            ):
                title, url = title.strip(), url.strip()
                if url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append({"title": title, "link": url, "snippet": "", "position": i})
                    mentioned_sources.append({"source_name": title, "url": url, "relevance": ""})

        # Strip [SOURCE:...] tags from the displayed analysis
        analysis_clean = re.sub(r"\[SOURCE:[^\]]+\]", "", analysis).strip()

        return search_results, analysis_clean, mentioned_sources

    except Exception as e:
        logger.error(f"OpenAI web search error: {e}", exc_info=True)
        return [], f"Web search error: {str(e)}", []

# Full pipeline
def analyze_news(statement: str, search_query: Optional[str] = None) -> dict:
    lang_code, lang_name = detect_language(statement)
    was_translated = lang_code != "en"

    english_text = (
        translate_text(statement, "en", lang_code)
        if was_translated else statement
    )

    label, label_text, prob, confidence = run_model(english_text)
    search_results, analysis, mentioned_sources = perform_web_search(english_text, search_query)

    if was_translated and analysis:
        analysis = translate_text(analysis, lang_code, "en")

    return {
        "statement":          statement,
        "detected_language":  lang_name,
        "language_code":      lang_code,
        "was_translated":     was_translated,
        "label":              label,
        "label_text":         label_text,
        "probability":        prob,
        "confidence":         confidence,
        "search_results":     search_results,
        "analysis":           analysis,
        "mentioned_sources":  mentioned_sources,
    }

# UI helpers
def format_analysis(text: str) -> str:
    text = re.sub(r"\[SOURCE:[^\]]+\]", "", text)
    text = re.sub(r"(\d+\.)\s*", r"\n\n\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# Streamlit app
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 0.4rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.fake-label {
    background-color: #ff4444;
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 6px;
    font-weight: bold;
    font-size: 1.1rem;
    display: inline-block;
}
.real-label {
    background-color: #28a745;
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 6px;
    font-weight: bold;
    font-size: 1.1rem;
    display: inline-block;
}
.source-card {
    background-color: #f0f4f8;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 0.9rem;
    border-left: 4px solid #1f77b4;
}
.source-card a {
    color: #1f77b4;
    font-weight: 600;
    text-decoration: none;
}
.source-card a:hover { text-decoration: underline; }
.source-snippet { color: #444; font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered News Verification with Web Search</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        Detects fake news using a trained ML model combined with live web search and GPT-4o analysis.

        **Steps:**
        1. Enter a news statement
        2. ML model classifies the text
        3. Web search finds related sources
        4. GPT-4o cross-verifies with sources
        5. Results shown with clickable sources
        """)

    st.header("Enter News Statement")
    news_statement = st.text_area(
        "Paste or type the news statement to verify:",
        height=150,
        placeholder="Example: 'Scientists discover cure for cancer in 2024'",
    )

    with st.expander("🔧 Advanced Options"):
        custom_search = st.text_input(
            "Custom search query (optional):",
            placeholder="Leave empty to auto-generate",
        )

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        analyze = st.button("Analyze News", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if analyze and news_statement:
        with st.spinner("Analyzing statement and searching the web…"):
            try:
                result = analyze_news(
                    news_statement,
                    custom_search if custom_search else None,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        st.success("✅ Analysis Complete!")
        st.header("Analysis Results")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🤖 ML Model Classification")
            m1, m2 = st.columns(2)
            m1.metric("Verdict", result["label_text"])
            m2.metric("Probability (Fake)", f"{result['probability'] * 100:.1f}%")

            if result["label_text"] == "FAKE":
                st.markdown('<div class="fake-label">⚠️ LIKELY FAKE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="real-label">✓ LIKELY REAL</div>', unsafe_allow_html=True)

            st.write("**Fake News Probability:**")
            st.progress(result["probability"])
            st.metric("Model Reliability", result["confidence"])

            if result["was_translated"]:
                st.info(f"🌐 Detected language: **{result['detected_language']}** — translated to English for analysis.")

        with col_right:
            st.subheader("🔍 AI Web Search Verification")
            analysis_raw = result["analysis"]
            if analysis_raw:
                st.markdown(format_analysis(analysis_raw))
            else:
                st.warning("No AI analysis available.")

        st.markdown("---")

        mentioned   = result["mentioned_sources"]
        all_results = result["search_results"]

        if mentioned:
            st.header("📰 Sources Referenced in Analysis")
            for src in mentioned:
                st.markdown(
                    f'<div class="source-card">'
                    f'<a href="{src["url"]}" target="_blank">{src["source_name"]}</a>'
                    f'<div class="source-snippet">{src["relevance"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if all_results:
            with st.expander(f"🌐 All Web Search Results ({len(all_results)})"):
                for r in all_results:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span style="color:#888;font-size:0.8rem;">#{r["position"]}</span> '
                        f'<a href="{r["link"]}" target="_blank">{r["title"]}</a>'
                        f'<div class="source-snippet">{r["snippet"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.header("📋 Final Assessment")
        s1, s2 = st.columns(2)

        with s1:
            st.markdown("### 🤖 ML Model")
            st.markdown(f"""
- **Verdict:** {result['label_text']}
- **Fake probability:** {result['probability'] * 100:.1f}%
- **Reliability:** {result['confidence']}
""")

        with s2:
            st.markdown("### 🔍 Web Verification")
            st.markdown(f"""
- **Sources matched:** {len(mentioned)}
- **Total results fetched:** {len(all_results)}
- **Analysis:** {"Available" if analysis_raw else "Unavailable"}
""")

        st.info(
            "**Recommendation:** Use both results together. "
            "If both ML model and web analysis agree, confidence is higher. "
            "Always cross-check critical claims with trusted outlets."
        )

        st.markdown("---")
        st.header("💾 Export Results")
        export = {
            "timestamp":       datetime.now().isoformat(),
            "statement":       news_statement,
            "verdict":         result["label_text"],
            "probability":     result["probability"],
            "confidence":      result["confidence"],
            "analysis":        analysis_raw,
            "mentioned_sources": [
                {"name": s["source_name"], "url": s["url"], "relevance": s["relevance"]}
                for s in mentioned
            ],
            "all_search_results": [
                {"title": r["title"], "link": r["link"], "snippet": r["snippet"]}
                for r in all_results
            ],
        }
        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(export, indent=2),
            file_name=f"fake_news_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    elif analyze:
        st.warning("Please enter a news statement to analyze.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;'>"
        "<small>Disclaimer: AI-assisted analysis only. Always verify critical claims through multiple reliable sources.</small>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
