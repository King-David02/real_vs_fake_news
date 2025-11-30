# import streamlit as st
# import requests

# st.title("Fake News Classifier ")
# st.write("Enter a news statement to predict whether it is real or fake.")

# # User input
# user_input = st.text_area("News Statement:")

# if st.button("Predict") and user_input:
#     # Prepare payload for API
#     payload = {"statement": user_input}
    
#     # Send POST request to FastAPI
#     response = requests.post("http://127.0.0.1:8000/predict", json=payload)

#     if response.status_code == 200:
#         data = response.json()
#         st.write("---")
#         st.write(f"**Prediction:** {'Fake' if data['label'] == 0 else 'Real'}")
#         st.write(f"**Probability:** {data['probability']:.2f}")
#     else:
#         st.error(f"API request failed with status code {response.status_code}")


import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

if 'statement' not in st.session_state:
    st.session_state.statement = ""

st.title("Fake News Detector")
st.markdown("""
Welcome! This tool helps you identify whether a news statement might be fake or real.
Simply paste any news headline or statement below and click **Analyze**.
""")

st.markdown("---")

statement = st.text_area(
    "Enter a news statement to check:",
    value=st.session_state.statement,
    height=150,
    placeholder="Example: Scientists discover cure for all diseases overnight...",
    help="Paste any news headline, article excerpt, or statement you want to verify",
    key="statement_input"
)

st.session_state.statement = statement

with st.expander("Try these example statements"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Example 1: Fake News", key="fake_example"):
            st.session_state.statement = "Breaking: Aliens land in New York City, confirm they invented pizza"
            st.rerun()
    
    with col2:
        if st.button("Example 2: Real News", key="real_example"):
            st.session_state.statement = "Stock market closes with modest gains amid economic uncertainty"
            st.rerun()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000 /predict")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Analyze Statement", use_container_width=True, type="primary")

if analyze_button:
    if st.session_state.statement.strip():
        with st.spinner("Analyzing the statement..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"statement": st.session_state.statement},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    label = result['label']
                    probability = result['probability']
                    
                    if label == 1:
                        st.success("**This statement appears to be REAL NEWS**")
                        confidence = probability * 100
                    else:
                        st.error("**This statement is likely FAKE NEWS**")
                        confidence = (1 - probability) * 100
                    
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    st.progress(confidence / 100)
                    
                    st.info("""
                    **What does this mean?**
                    - This is a prediction based on machine learning analysis
                    - Higher confidence means the model is more certain
                    - Always verify important news from multiple trusted sources
                    """)
                    
                    st.caption("This tool is for educational purposes. Always fact-check important information from reliable sources.")
                    
                else:
                    st.error(f"Error: Unable to analyze (Status code: {response.status_code})")
                    
            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the analysis service. Please check if the API is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a statement to analyze")

st.markdown("---")
st.markdown("""
### How to use this tool:
1. Paste or type a news statement in the text box above
2. Click the **Analyze Statement** button
3. Review the results and confidence level
4. Remember to verify important news from multiple trusted sources

### About
This tool uses machine learning to analyze text patterns and predict whether a statement might be fake news.
It's designed to help you think critically about information you encounter online.
""")