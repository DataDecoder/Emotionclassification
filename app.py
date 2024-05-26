import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

# Load the saved models
lg = pickle.load(open('logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion

# Streamlit application
st.title("Emotion Analysis")

comment = st.text_area("Enter your comment:")

if st.button("Analyze"):
    if comment:
        predicted_emotion = predict_emotion(input_text=comment)
        st.write(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.write("Please enter a comment.")