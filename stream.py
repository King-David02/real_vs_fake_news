import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fake-label {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .real-label {
        background-color: #44ff44;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .search-result {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_news(statement: str, search_query: str = None):
    """Call the FastAPI backend to check news"""
    try:
        payload = {"statement": statement}
        if search_query:
            payload["search_query"] = search_query
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered News Verification with Web Search</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses machine learning and web search to detect fake news.
        
        **Features:**
        - ML-based classification
        - Real-time web search
        - Source verification
        - Credibility analysis
        """)
        
        st.header("📊 How it works")
        st.write("""
        1. Enter a news statement
        2. AI model analyzes the text
        3. Web search finds related sources
        4. Results are cross-verified
        5. Get a verdict with sources
        """)
        
        st.header("⚙️ Settings")
        api_status = st.empty()
        
        # Check API status
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                api_status.success("✅ API Connected")
            else:
                api_status.error("❌ API Error")
        except:
            api_status.error("❌ API Offline")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter News Statement")
        news_statement = st.text_area(
            "Paste or type the news statement you want to verify:",
            height=150,
            placeholder="Example: 'Scientists discover cure for cancer in 2024'"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            custom_search = st.text_input(
                "Custom search query (optional):",
                placeholder="Leave empty to auto-generate"
            )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_button = st.button("Analyze News", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("Clear", use_container_width=True)
    
    with col2:
        st.header("💡 Example Statements")
        examples = [
            "NASA announces discovery of alien life on Mars",
            "COVID-19 vaccines contain microchips for tracking",
            "Biden wins 2024 presidential election",
            "New study shows coffee reduces cancer risk"
        ]
        
        selected_example = st.selectbox("Try an example:", [""] + examples)
        if selected_example:
            news_statement = selected_example
            st.rerun()
    
    # Clear functionality
    if clear_button:
        st.rerun()
    
    # Analysis
    if analyze_button and news_statement:
        with st.spinner("Analyzing statement and searching the web..."):
            result, error = check_news(
                news_statement, 
                custom_search if custom_search else None
            )
            
            if error:
                st.error(f"{error}")
            elif result:
                st.success("Analysis Complete!")
                
                # Display results
                st.header("Analysis Results")
                
                # Create two columns for model results and AI analysis
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("ML Model Classification")
                    
                    # Prediction metrics
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        st.metric("Model Verdict", result['label_text'])
                    
                    with col_pred2:
                        probability_pct = result['probability'] * 100
                        st.metric("Confidence Score", f"{probability_pct:.1f}%")
                    
                    # Visual indicator for model
                    if result['label_text'] == "FAKE":
                        st.markdown(
                            f'<div class="fake-label">⚠️ MODEL: LIKELY FAKE</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="real-label">✓ MODEL: LIKELY REAL</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Progress bar for probability
                    st.write("**Fake News Probability:**")
                    st.progress(result['probability'])
                    
                    st.metric("Model Reliability", result['confidence'])
                    
                    # Model explanation
                    st.info(f"""
                    **How the model works:**
                    - Analyzes text patterns and linguistic features
                    - Trained on thousands of verified news articles
                    - Probability score: {probability_pct:.1f}% likelihood of being fake
                    """)
                
                with col_right:
                    st.subheader("AI Web Search Verification")
                    
                    # AI Analysis
                    if result.get('analysis'):
                        st.markdown("**AI Analysis based on web sources:**")
                        st.success(result['analysis'])
                    else:
                        st.warning("No AI analysis available")
                    
                    # Source count
                    if result.get('search_results'):
                        st.metric("Sources Found", len(result['search_results']))
                        st.info("""
                        **How AI verification works:**
                        - Searches the web for related information
                        - Cross-references multiple credible sources
                        - Provides context and fact-checking insights
                        """)
                    else:
                        st.warning("No web sources found for verification")
                
                st.markdown("---")
                # Search Results Section
                st.header("Web Sources & Evidence")
                
                if result.get('search_results'):
                    st.write(f"**Found {len(result['search_results'])} relevant sources from the web:**")
                    
                    for idx, source in enumerate(result['search_results'], 1):
                        with st.container():
                            st.markdown(f"""
                                <div class="search-result">
                                    <h4>{idx}. {source['title']}</h4>
                                    <p>{source['snippet']}</p>
                                    <a href="{source['link']}" target="_blank">🔗 Read full article</a>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No web search results available for cross-verification.")
                # Comparison Summary
                st.header("📋 Final Assessment")
                
                col_summary1, col_summary2 = st.columns(2)
                
                with col_summary1:
                    st.markdown("### Machine Learning Model")
                    st.markdown(f"""
                    - **Verdict:** {result['label_text']}
                    - **Confidence:** {result['probability'] * 100:.1f}%
                    - **Reliability:** {result['confidence']}
                    """)
                
                with col_summary2:
                    st.markdown("### 🔍 AI Web Verification")
                    if result.get('search_results'):
                        st.markdown(f"""
                        - **Sources Checked:** {len(result['search_results'])}
                        - **Analysis:** Available
                        - **Cross-Referenced:** Yes
                        """)
                    else:
                        st.markdown("""
                        - **Sources Checked:** 0
                        - **Analysis:** Limited
                        - **Cross-Referenced:** No
                        """)
                
                # Overall recommendation
                st.info("""
                **Recommendation:** 
                - Use both the ML model prediction AND the AI web verification together
                - Check the provided sources to form your own opinion
                - Be especially cautious if both methods indicate "FAKE"
                - Consider the confidence scores and source quality
                """)
                
                st.markdown("---")
                st.header("💾 Export Results")
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "statement": news_statement,
                    "verdict": result['label_text'],
                    "probability": result['probability'],
                    "confidence": result['confidence'],
                    "analysis": result.get('analysis', ''),
                    "sources": [
                        {"title": s['title'], "link": s['link'], "snippet": s['snippet']}
                        for s in result.get('search_results', [])
                    ]
                }
                
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"fake_news_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    elif analyze_button:
        st.warning("Please enter a news statement to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p><small>Disclaimer: This tool provides AI-assisted analysis. Always verify important claims through multiple reliable sources.</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



