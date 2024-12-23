import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

ps = PorterStemmer()

# Load Punkt tokenizer model from the pickle file
try:
    with open('english.pickle', 'rb') as file:
        punkt_tokenizer = PunktSentenceTokenizer(pickle.load(file))
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

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text, tokenizer=punkt_tokenizer)  # Tokenize using loaded Punkt model

    y = []
    for i in text:
        if i.isalnum():  # Keep alphanumeric tokens
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:  # Remove stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)

# Load model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
