import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer

ps = PorterStemmer()

# Load Punkt tokenizer model from the pickle file
try:
    with open('english.pickle', 'rb') as file:
        punkt_tokenizer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading Punkt tokenizer: {e}")
    st.stop()

# Define stopwords list
stop_words = """
i me my myself we our ours ourselves you you're you've you'll you'd your yours 
yourself yourselves he him his himself she she's her hers herself it it's its 
itself they them their theirs themselves what which who whom this that that'll 
these those am is are was were be been being have has had having do does did 
doing a an the and but if or because as until while of at by for with about 
against between into through during before after above below to from up down 
in out on off over under again further then once here there when where why how 
all any both each few more most other some such no nor not only own same so 
than too very s t can will just don don't should should've now d ll m o re ve 
y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn 
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't 
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split()

import re

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using Punkt tokenizer
    tokens = punkt_tokenizer.tokenize(text)
    # st.write("Tokens:", tokens)  # Check tokens

    # Keep alphanumeric tokens and split punctuation
    y = []
    for token in tokens:
        # Split punctuation from words using regex
        words = re.findall(r'\b\w+\b', token)
        y.extend(words)
    
    # st.write("Tokens after keeping meaningful words:", y)  # Check after splitting punctuation

    # Remove stopwords and punctuation
    y = [word for word in y if word not in stop_words]

    # st.write("Tokens after stopword removal:", y)  # Check after removing stopwords

    # Apply stemming
    y = [ps.stem(word) for word in y]

    # st.write("Tokens after stemming:", y)  # Check after stemming

    return " ".join(y)


# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

# Input box for SMS
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        # st.write("**Transformed Text:**", transformed_sms)  # Display transformed text

        # Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        # st.write("**Vectorized Input Shape:**", vector_input.shape)  # Display vectorized input shape

        # Debugging the top features in the vectorizer
        try:
            top_features = tfidf.get_feature_names_out()[:20]
            # st.write("**Top Features in Vectorizer:**", top_features)  # Display top features
        except Exception as e:
            st.write("**Error Retrieving Top Features:**", e)

        # Predict the result
        try:
            result = model.predict(vector_input)[0]
            prediction_probabilities = model.predict_proba(vector_input)
            st.write("**Prediction Probabilities:**", prediction_probabilities)  # Display probabilities
            st.write("**Raw Prediction:**", result)  # Display raw prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
