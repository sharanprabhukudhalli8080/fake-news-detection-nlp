import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below to check whether it is Real or Fake.")

# Show model classes (for debugging)
st.write("Model Classes:", model.classes_)

user_input = st.text_area("Enter News Text", height=200)

if st.button("Check"):
    if user_input.strip() != "":
        # Vectorize input
        text_vec = vectorizer.transform([user_input])

        # Predict class
        prediction = model.predict(text_vec)[0]

        # Get probabilities
        probs = model.predict_proba(text_vec)[0]

        # Map class -> probability safely
        class_prob_map = dict(zip(model.classes_, probs))

        # Try common mappings
        fake_prob = None
        real_prob = None

        # Case 1: labels are strings ("FAKE", "REAL")
        for key in class_prob_map:
            if str(key).lower() in ["fake", "0"]:
                fake_prob = class_prob_map[key]
            if str(key).lower() in ["real", "1"]:
                real_prob = class_prob_map[key]

        # Fallback if numeric but reversed
        if fake_prob is None or real_prob is None:
            # Assume smaller prob = fake? No — show both safely
            fake_prob = min(class_prob_map.values())
            real_prob = max(class_prob_map.values())

        # Show result
        if fake_prob > real_prob:
            st.error(f"🛑 Fake News")
        else:
            st.success(f"✅ Real News")

        # Show probabilities (debug + transparency)
        st.write(f"Fake Probability: **{fake_prob:.3f}**")
        st.write(f"Real Probability: **{real_prob:.3f}**")

    else:
        st.warning("Please enter some text.")
