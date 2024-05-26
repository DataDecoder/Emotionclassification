# Emotion Analysis Web App
This repository contains a Streamlit-based web application for emotion analysis. The application uses a pre-trained Logistic Regression model to predict the emotion expressed in user-inputted text.

### Features
- Text Preprocessing: Cleans and preprocesses the input text using NLTK.
- Emotion Prediction: Uses a trained Logistic Regression model to predict emotions.
- Interactive Interface: Provides a user-friendly interface with Streamlit for easy interaction.

### How to Use
Clone the repository: 
git clone https://github.com/your-username/emotion-analysis-webapp.git
cd emotion-analysis-webapp

### Install the required packages:
pip install -r requirements.txt

### Ensure NLTK stopwords are downloaded:
import nltk
nltk.download('stopwords')

### Run the Streamlit app:
streamlit run app.py

### Access the application: Open the provided local URL in your web browser.
<img width="592" alt="image" src="https://github.com/DataDecoder/Emotionclassification/assets/72354914/52b6359b-37a4-4467-aa11-674b7f2c168c">

### Files
- app.py: The main Streamlit application file.
- logistic_regression.pkl: Pre-trained Logistic Regression model.
- tfidf_vectorizer.pkl: TF-IDF Vectorizer for text transformation.
- label_encoder.pkl: Label Encoder for decoding predicted labels.



