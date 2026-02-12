import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below to check whether it is Real or Fake.")

user_input = st.text_area("Enter News Text", height=200)

if st.button("Check"):
    if user_input.strip() != "":
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]

        fake_index = list(model.classes_).index(0)  # change if Fake=1
        real_index = list(model.classes_).index(1)

        fake_prob = probs[fake_index]
        real_prob = probs[real_index]

        if fake_prob > real_prob:
            st.error(f"🛑 Fake News (Confidence: {fake_prob:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {real_prob:.2f})")
    else:
        st.warning("Please enter some text.")
