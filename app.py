import streamlit as st
import pickle as pk
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
tfidf = pk.load(open('vectorizer.pkl','rb'))
model = pk.load(open('model.pkl','rb'))


def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    transform_sms = transform(input_sms)
    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
