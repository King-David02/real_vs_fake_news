import re
import json
import requests
from datetime import datetime
import streamlit as st

# Page config 
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

API_BASE_URL = "http://localhost:8000"

# API call 
def check_news(statement: str, search_query: str = None):
    try:
        payload = {"statement": statement}
        if search_query:
            payload["search_query"] = search_query

        resp = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API Error {resp.status_code}: {resp.text}"

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000"
    except Exception as e:
        return None, f"Error: {e}"

# Helpers 
def format_analysis(text: str) -> str:
    """Clean up analysis text and hide raw [SOURCE:] tags."""
    text = re.sub(r"\[SOURCE:[^\]]+\]", "", text)          # remove raw tags
    text = re.sub(r"(\d+\.)\s*", r"\n\n\1 ", text)         # newline before numbered items
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# Main 
def main():
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered News Verification with Web Search</div>', unsafe_allow_html=True)

    # Sidebar
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

        st.header("⚙️ API Status")
        try:
            r = requests.get(f"{API_BASE_URL}/", timeout=5)
            if r.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except Exception:
            st.error("❌ API Offline")

    # Input
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

    # Results 
    if analyze and news_statement:
        with st.spinner("Analyzing statement and searching the web…"):
            result, error = check_news(
                news_statement,
                custom_search if custom_search else None,
            )

        if error:
            st.error(error)
            return

        st.success("✅ Analysis Complete!")
        st.header("Analysis Results")

        col_left, col_right = st.columns(2)

        # Left: ML model
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

            if result.get("was_translated"):
                st.info(f"🌐 Detected language: **{result.get('detected_language')}** — translated to English for analysis.")

        # Right: AI web verification 
        with col_right:
            st.subheader("🔍 AI Web Search Verification")

            analysis_raw = result.get("analysis", "")
            if analysis_raw:
                source_count = len(re.findall(r"\[SOURCE:[^\]]+\]", analysis_raw))
                st.markdown(format_analysis(analysis_raw))
                if source_count:
                    st.success(f"✅ {source_count} source(s) referenced in analysis")
            else:
                st.warning("No AI analysis available.")

        st.markdown("---")

        # Sources
        mentioned = result.get("mentioned_sources", [])
        all_results = result.get("search_results", [])

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

        # Final summary 
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
            n = len(mentioned)
            st.markdown(f"""
- **Sources matched:** {n}
- **Total results fetched:** {len(all_results)}
- **Analysis:** {"Available" if analysis_raw else "Unavailable"}
""")

        st.info(
            "**Recommendation:** Use both results together. "
            "If both ML model and web analysis agree, confidence is higher. "
            "Always cross-check critical claims with trusted outlets."
        )

        st.markdown("---")

        # Export
        st.header("💾 Export Results")
        export = {
            "timestamp": datetime.now().isoformat(),
            "statement": news_statement,
            "verdict": result["label_text"],
            "probability": result["probability"],
            "confidence": result["confidence"],
            "analysis": analysis_raw,
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
