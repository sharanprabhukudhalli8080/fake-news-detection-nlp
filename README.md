# 📰 Fake News Detection System (End-to-End NLP Web App)

An end-to-end Natural Language Processing (NLP) project that classifies news articles as **Real** or **Fake** using TF-IDF and Machine Learning, deployed as a live Streamlit web application.

🔴 **Live Demo:** https://fake-news-detection-nlp-39a9v7tpptqbzeskmt9kbs.streamlit.app
📦 **GitHub Repo:** https://github.com/sharanprabhukudhalli8080/fake-news-detection-nlp

---

## 🚀 Key Features
- Text preprocessing (lowercasing, stopword removal, cleaning)
- Feature extraction using **TF-IDF Vectorization**
- Machine Learning model:
  - Logistic Regression
- Binary classification:
  - 0 → Fake News  
  - 1 → Real News
- Confidence score for each prediction
- Deployed as a **Streamlit Web App**
- Real-time news classification through text input

---

## 🧠 Tech Stack
- Python  
- Pandas  
- NLTK  
- Scikit-learn  
- TF-IDF  
- Joblib  
- Streamlit  

---

## 📊 Problem Statement
Fake news spreads rapidly through digital platforms and can influence public opinion, elections, and social stability.  
This project aims to automatically classify news articles as **Real** or **Fake** using Natural Language Processing and Machine Learning to support content verification.

---

## 🏗 Project Architecture

fake-news-detection-nlp
│
├── app.py
  Streamlit web application

├── fake_news_model.pkl
  Trained Machine Learning classification model

├── tfidf_vectorizer.pkl
  TF-IDF feature extractor

├── requirements.txt
  Project dependencies

├── README.md
  Project documentation

├── LICENSE
  MIT License

└── .gitignore
  Ignored files and folders for Git version control

  
---

## ▶ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sharanprabhukudhalli8080/fake-news-detection-nlp.git
cd fake-news-detection-nlp


---

## Install dependencies
pip install -r requirements.txt


---

## Run the Streamlit app

streamlit run app.py

---

## 🌐 Web App Workflow

1.User pastes a news article into the text box.

2.Text is cleaned and transformed using TF-IDF.

3.The trained Logistic Regression model predicts:

  -Fake News or Real News

4.A confidence score is displayed.

5.Output is shown instantly on the web interface.

---

## 📈 Model Performance

-Accuracy: ~95%+

-High precision and recall on both fake and real classes

-Robust generalization on unseen news articles






















