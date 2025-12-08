# import streamlit as st
# import requests
# import os

# import os
# API_URL = os.getenv("API_URL", "http://localhost:8000")

# st.set_page_config(
#     page_title="Fake News Detector",
#     page_icon="🔍",
#     layout="centered"
# )

# if 'statement' not in st.session_state:
#     st.session_state.statement = ""

# st.title("Fake News Detector")
# st.markdown("""
# Welcome! This tool helps you identify whether a news statement might be fake or real.
# Simply paste any news headline or statement below and click **Analyze**.
# """)

# st.markdown("---")

# statement = st.text_area(
#     "Enter a news statement to check:",
#     value=st.session_state.statement,
#     height=150,
#     placeholder="Example: Scientists discover cure for all diseases overnight...",
#     help="Paste any news headline, article excerpt, or statement you want to verify",
#     key="statement_input"
# )

# st.session_state.statement = statement

# with st.expander("Try these example statements"):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("Example 1: Fake News", key="fake_example"):
#             st.session_state.statement = "Breaking: Aliens land in New York City, confirm they invented pizza"
#             st.rerun()
    
#     with col2:
#         if st.button("Example 2: Real News", key="real_example"):
#             st.session_state.statement = "Stock market closes with modest gains amid economic uncertainty"
#             st.rerun()

# API_URL = "http://api:8000/predict"

# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     analyze_button = st.button("Analyze Statement", use_container_width=True, type="primary")

# if analyze_button:
#     if st.session_state.statement.strip():
#         with st.spinner("Analyzing the statement..."):
#             try:
#                 response = requests.post(
#                     API_URL,
#                     json={"statement": st.session_state.statement},
#                     timeout=30
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
                    
#                     st.markdown("---")
#                     st.subheader("Analysis Results")
                    
#                     label = result['label']
#                     probability = result['probability']
                    
#                     if label == 1:
#                         st.success("**This statement appears to be REAL NEWS**")
#                         confidence = probability * 100
#                     else:
#                         st.error("**This statement is likely FAKE NEWS**")
#                         confidence = (1 - probability) * 100
                    
#                     st.metric("Confidence Level", f"{confidence:.1f}%")
#                     st.progress(confidence / 100)
                    
#                     st.info("""
#                     **What does this mean?**
#                     - This is a prediction based on machine learning analysis
#                     - Higher confidence means the model is more certain
#                     - Always verify important news from multiple trusted sources
#                     """)
                    
#                     st.caption("This tool is for educational purposes. Always fact-check important information from reliable sources.")
                    
#                 else:
#                     st.error(f"Error: Unable to analyze (Status code: {response.status_code})")
                    
#             except requests.exceptions.Timeout:
#                 st.error("The request timed out. Please try again.")
#             except requests.exceptions.ConnectionError:
#                 st.error("Cannot connect to the analysis service. Please check if the API is running.")
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#     else:
#         st.warning("Please enter a statement to analyze")

# st.markdown("---")
# st.markdown("""
# ### How to use this tool:
# 1. Paste or type a news statement in the text box above
# 2. Click the **Analyze Statement** button
# 3. Review the results and confidence level
# 4. Remember to verify important news from multiple trusted sources

# ### About
# This tool uses machine learning to analyze text patterns and predict whether a statement might be fake news.
# It's designed to help you think critically about information you encounter online.
# """)




import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL")
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Fake News Detector with Source Verification",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'statement' not in st.session_state:
    st.session_state.statement = ""

# Header
st.title("🔍 Fake News Detector with Source Verification")
st.markdown("""
Welcome! This advanced tool combines **AI analysis** with **real-time web search** to help you identify 
whether a news statement might be fake or real. Enter any news headline or claim below to get started.
""")

st.markdown("---")

# Main input area
statement = st.text_area(
    "Enter a news statement to verify:",
    value=st.session_state.statement,
    height=150,
    placeholder="Example: Breaking news about recent scientific discoveries, political events, or viral claims...",
    help="Paste any news headline, article excerpt, or claim you want to verify",
    key="statement_input"
)

st.session_state.statement = statement

# Example statements
with st.expander("📝 Try these example statements"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 Fake Example", key="fake_example", use_container_width=True):
            st.session_state.statement = "Breaking: Scientists discover that drinking bleach cures all diseases, government trying to hide this information from public"
            st.rerun()
    
    with col2:
        if st.button("✅ Real Example", key="real_example", use_container_width=True):
            st.session_state.statement = "Stock market experiences volatility as Federal Reserve announces new interest rate decision"
            st.rerun()
    
    with col3:
        if st.button("❓ Ambiguous Example", key="ambiguous_example", use_container_width=True):
            st.session_state.statement = "New study suggests that coffee may have health benefits when consumed in moderation"
            st.rerun()

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🔎 Analyze Statement", 
        use_container_width=True, 
        type="primary"
    )

# Analysis logic
if analyze_button:
    if st.session_state.statement.strip():
        
        # Create tabs for different result sections
        with st.spinner("🔄 Analyzing the statement and searching for sources..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"statement": st.session_state.statement},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    
                    # Create tabs for organized results
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Quick Summary", 
                        "🤖 AI Analysis", 
                        "🔗 Sources Found",
                        "✅ Final Verdict"
                    ])
                    
                    # TAB 1: Quick Summary
                    with tab1:
                        st.subheader("Quick Summary")
                        
                        verdict = result['final_conclusion']['verdict']
                        confidence = result['final_conclusion']['confidence']
                        
                        # Display verdict with appropriate styling
                        if verdict in ['VERIFIED REAL', 'LIKELY REAL']:
                            st.success(f"### ✅ {verdict}")
                            verdict_color = "green"
                        elif verdict in ['LIKELY FAKE']:
                            st.error(f"### 🚨 {verdict}")
                            verdict_color = "red"
                        elif verdict in ['CONFLICTING SIGNALS', 'NEEDS VERIFICATION']:
                            st.warning(f"### ⚠️ {verdict}")
                            verdict_color = "orange"
                        else:
                            st.info(f"### ℹ️ {verdict}")
                            verdict_color = "blue"
                        
                        # Metrics row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence Level", confidence)
                        with col2:
                            st.metric("Sources Found", result['sources_count'])
                        with col3:
                            model_conf = result['probability'] if result['label'] == 1 else (1 - result['probability'])
                            st.metric("AI Confidence", f"{model_conf*100:.1f}%")
                        
                        st.markdown("---")
                        st.markdown(f"**Explanation:** {result['final_conclusion']['explanation']}")
                    
                    # TAB 2: AI Analysis
                    with tab2:
                        st.subheader("🤖 AI Model Analysis")
                        
                        label = result['label']
                        probability = result['probability']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if label == 1:
                                st.success("**Model Prediction: REAL NEWS**")
                                confidence_pct = probability * 100
                            else:
                                st.error("**Model Prediction: FAKE NEWS**")
                                confidence_pct = (1 - probability) * 100
                        
                        with col2:
                            st.metric("Model Confidence", f"{confidence_pct:.1f}%")
                        
                        st.progress(confidence_pct / 100)
                        
                        # Show extracted claims
                        st.markdown("#### 📋 Key Claims Extracted for Search")
                        if result['claims_extracted']:
                            for i, claim in enumerate(result['claims_extracted'], 1):
                                st.markdown(f"{i}. *{claim}*")
                        else:
                            st.info("No specific claims could be extracted. Searched using the full statement.")
                        
                        st.info("""
                        **About the AI Model:**
                        - Trained on thousands of real and fake news articles
                        - Analyzes text patterns, writing style, and linguistic features
                        - Provides probability score indicating confidence level
                        """)
                    
                    # TAB 3: Sources Found
                    with tab3:
                        st.subheader("🔗 Sources Found on the Web")
                        
                        if not result['search_enabled']:
                            st.warning("⚠️ Web search is currently disabled. Only AI analysis is available.")
                            st.info("To enable source verification, configure the NEWS_API_KEY in your environment variables.")
                        
                        elif result['sources_count'] == 0:
                            st.info("🔍 No sources found online for this claim.")
                            st.markdown("""
                            This could mean:
                            - The news is very recent and hasn't been widely reported yet
                            - The claim is not covered by major news outlets
                            - The claim might be completely fabricated
                            """)
                        
                        else:
                            st.success(f"Found {result['sources_count']} related sources")
                            
                            # Display sources
                            for i, source in enumerate(result['sources_found'], 1):
                                credibility = source['credibility']
                                
                                # Determine credibility indicator
                                if credibility >= 0.85:
                                    cred_icon = "🌟🌟🌟"
                                    cred_label = "High Credibility"
                                    cred_color = "green"
                                elif credibility >= 0.70:
                                    cred_icon = "🌟🌟"
                                    cred_label = "Good Credibility"
                                    cred_color = "blue"
                                elif credibility >= 0.50:
                                    cred_icon = "🌟"
                                    cred_label = "Moderate Credibility"
                                    cred_color = "orange"
                                else:
                                    cred_icon = "⚠️"
                                    cred_label = "Low Credibility"
                                    cred_color = "red"
                                
                                with st.expander(f"{cred_icon} **Source {i}:** {source['title'][:100]}..."):
                                    st.markdown(f"**Source:** {source['source']}")
                                    st.markdown(f"**Credibility:** :{cred_color}[{cred_label} ({credibility:.2f})]")
                                    
                                    if source['description']:
                                        st.markdown(f"**Description:** {source['description']}")
                                    
                                    if source['published_at']:
                                        st.markdown(f"**Published:** {source['published_at'][:10]}")
                                    
                                    st.markdown(f"**Searched for:** *{source['claim_searched']}*")
                                    st.markdown(f"[🔗 Read Full Article]({source['url']})")
                            
                            # Average credibility
                            if result['sources_found']:
                                avg_cred = sum(s['credibility'] for s in result['sources_found']) / len(result['sources_found'])
                                st.markdown("---")
                                st.metric("Average Source Credibility", f"{avg_cred:.2f}")
                    
                    # TAB 4: Final Verdict
                    with tab4:
                        st.subheader("✅ Final Verdict & Recommendation")
                        
                        verdict = result['final_conclusion']['verdict']
                        
                        # Display final verdict with large emphasis
                        if verdict in ['VERIFIED REAL', 'LIKELY REAL']:
                            st.success(f"# ✅ {verdict}")
                        elif verdict in ['LIKELY FAKE']:
                            st.error(f"# 🚨 {verdict}")
                        elif verdict in ['CONFLICTING SIGNALS', 'NEEDS VERIFICATION']:
                            st.warning(f"# ⚠️ {verdict}")
                        else:
                            st.info(f"# ℹ️ {verdict}")
                        
                        st.markdown("---")
                        
                        # Explanation
                        st.markdown("### 📝 Detailed Explanation")
                        st.write(result['final_conclusion']['explanation'])
                        
                        st.markdown("---")
                        
                        # Recommendation
                        st.markdown("### 💡 Recommendation")
                        st.info(result['final_conclusion']['recommendation'])
                        
                        st.markdown("---")
                        
                        # General advice
                        st.markdown("### 🎯 Best Practices for Fact-Checking")
                        st.markdown("""
                        1. **Cross-reference multiple sources** - Don't rely on a single source
                        2. **Check the source credibility** - Is it a known, reputable news organization?
                        3. **Look for evidence** - Does the article cite sources, studies, or data?
                        4. **Check the date** - Is this current news or an old story being recirculated?
                        5. **Be skeptical of sensational claims** - Extraordinary claims require extraordinary evidence
                        6. **Use fact-checking websites** - Sites like Snopes, FactCheck.org, and PolitiFact
                        """)
                    
                    # Footer disclaimer
                    st.markdown("---")
                    st.caption("""
                    ⚠️ **Disclaimer:** This tool is for educational and informational purposes only. 
                    While we combine AI analysis with real-time source verification, no automated system is 100% accurate. 
                    Always use critical thinking and verify important information from multiple trusted sources.
                    """)
                    
                else:
                    st.error(f"❌ Error: Unable to analyze (Status code: {response.status_code})")
                    if response.status_code == 500:
                        st.error("The server encountered an error. Please try again or check the API logs.")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ The request timed out. The service might be slow or unavailable. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the analysis service. Please check if the API is running at " + API_URL)
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
    else:
        st.warning("⚠️ Please enter a statement to analyze")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This Tool")
    
    st.markdown("""
    ### How it works:
    
    1. **AI Analysis** 🤖
       - Machine learning model analyzes text patterns
       - Provides initial fake/real prediction
    
    2. **Web Search** 🔍
       - Extracts key claims from your input
       - Searches for related news articles
       - Evaluates source credibility
    
    3. **Final Verdict** ✅
       - Combines AI prediction with source analysis
       - Provides confidence level and recommendation
    
    ### Source Credibility:
    - 🌟🌟🌟 High (0.85+): Reuters, AP, BBC
    - 🌟🌟 Good (0.70-0.84): CNN, NYT, WSJ
    - 🌟 Moderate (0.50-0.69): Unknown sources
    - ⚠️ Low (<0.50): Known unreliable sources
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Tips for Best Results:
    - Enter complete statements or headlines
    - Include specific claims or facts
    - Avoid very short or vague text
    - Check recent news for best source matching
    """)
    
    st.markdown("---")
    st.caption("Version 2.0 with Source Verification")