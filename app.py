import streamlit as st
import requests

st.title("Fake News Classifier ")
st.write("Enter a news statement to predict whether it is real or fake.")

# User input
user_input = st.text_area("News Statement:")

if st.button("Predict") and user_input:
    # Prepare payload for API
    payload = {"statement": user_input}
    
    # Send POST request to FastAPI
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        data = response.json()
        st.write("---")
        st.write(f"**Prediction:** {'Fake' if data['label'] == 0 else 'Real'}")
        st.write(f"**Probability:** {data['probability']:.2f}")
    else:
        st.error(f"API request failed with status code {response.status_code}")
