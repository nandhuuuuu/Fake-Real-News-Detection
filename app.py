
import streamlit as st
import pickle
import re
import pandas as pd

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.sidebar.title("Fake News Detection")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "Explain", "About"]
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

if page == "Home":

    st.title("Fake vs Real News Detection")

    st.write("""
    This project uses Machine Learning and Natural Language Processing
    to identify whether a news article is Fake or Real.

    The system analyses the text and predicts the category.
    """)

    st.subheader("Project Objectives")

    st.write("- Detect fake and real news")
    st.write("- Use NLP for text preprocessing")
    st.write("- Compare machine learning models")

elif page == "Predict":

    st.title("News Prediction")

    article = st.text_area("Enter your news article")

    if st.button("Predict"):

        if article.strip():

            cleaned = clean_text(article)
            vector = tfidf.transform([cleaned])

            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]

            classes = list(model.classes_)
            probs = dict(zip(classes, probabilities))

            st.subheader("Prediction")

            st.write("Result:", prediction)

            st.write("Confidence:", round(max(probabilities)*100, 2), "%")

            st.write("Fake Probability:",
                     round(probs.get('Fake', 0)*100, 2), "%")

            st.write("Real Probability:",
                     round(probs.get('Real', 0)*100, 2), "%")

        else:
            st.warning("Please enter an article")

elif page == "Explain":

    st.title("Explain Prediction")

    article = st.text_area("Enter an article to analyse")

    if st.button("Explain"):

        cleaned = clean_text(article)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        st.write("Predicted Category:", prediction)

        words = cleaned.split()

        st.write("Important words found in the article:")

        st.write(", ".join(words[:15]))

        st.caption(
            "These are words from the article. "
            "They are not individual causal explanations of the prediction."
        )

elif page == "About":

    st.title("About the Project")

    st.write("""
    Project: Fake vs Real News Detection

    Dataset: Fake and Real News Dataset

    Techniques used:
    - Text preprocessing
    - TF-IDF Vectorization
    - Logistic Regression
    - Naive Bayes
    - Random Forest

    Best performing model: Random Forest
    """)

    st.write("This project was developed using Python and Streamlit.")
