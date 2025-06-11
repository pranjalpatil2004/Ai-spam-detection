import streamlit as st
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧")
st.title("📧 AI Spam Mail Detector")
st.write("Enter your email text below and find out if it's spam or not!")

user_input = st.text_area("✉️ Enter Email Text Here:")

if st.button("Detect Spam"):
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)

    if prediction[0] == 1:
        st.error("🚫 This is a SPAM message.")
    else:
        st.success("✅ This is NOT a spam message.")
        