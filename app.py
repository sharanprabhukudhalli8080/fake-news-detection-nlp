import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below to check whether it is Real or Fake.")

# ====== LABEL MAPPING ======
FAKE_LABEL = 0
REAL_LABEL = 1
# ===========================

user_input = st.text_area("Enter News Text", height=200)

if st.button("Check"):
    if user_input.strip() != "":
        # Vectorize input
        text_vec = vectorizer.transform([user_input])

        # Predict class
        prediction = model.predict(text_vec)[0]

        # Get probabilities
        probs = model.predict_proba(text_vec)[0]

        fake_index = list(model.classes_).index(FAKE_LABEL)
        real_index = list(model.classes_).index(REAL_LABEL)

        fake_prob = probs[fake_index]
        real_prob = probs[real_index]

        # Show result
        if prediction == FAKE_LABEL:
            st.error(f"🛑 Fake News (Confidence: {fake_prob:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {real_prob:.2f})")

    else:
        st.warning("Please enter some text.")
