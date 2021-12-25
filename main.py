
from string import punctuation
from nltk.tokenize import word_tokenize
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
import sklearn
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
import unicodedata
from nltk import word_tokenize

app = FastAPI(
    title="Ticket Prediction API",
    description="A simple API that use NLP model to predict the group to which a ticket should be assigned basis the description",
    version="0.1",
)

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "model/ticket_classifier_model.pkl"), "rb"
) as f:
    model = joblib.load(f)

def clean_text(text, detect_translate=True, remove_stopwords=True, lemmatize_text=True):
    text=text.lower()
    text= re.sub(r"_x000D_",' ',text)
    text = re.sub(r'[\r|\n|\r\n]+', ' ',text)
    text = re.sub(r"received from:",' ',text)
    text = re.sub(r"from:",' ',text)
    text = re.sub(r"to:",' ',text)
    text = re.sub(r"subject:",' ',text)
    text = re.sub(r"sent:",' ',text)
    text = re.sub(r"ic:",' ',text)
    text = re.sub(r"cc:",' ',text)
    text = re.sub(r"bcc:",' ',text)
    text = re.sub(r"issue resolved.",' ', text)
    # Removing url
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    #Removing email 
    text = re.sub(r'\S+@\S+', '', text)
    text = text.replace("\\", ' ')
    # Removing numbers 
    text = re.sub(r'\d+','' ,text)
    # Removing accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove new line characters 
    text = re.sub(r'\n',' ',text)
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    #& 
    text = re.sub(r'&;?', 'and',text)
    # Remove HTML special entities (e.g. &amp;)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Remove characters beyond Readable formart by Unicode:
    text= ''.join(c for c in text if c <= '\uFFFF') 
    text = text.strip()
    # Removing special characters and\or digits    
    specialchar_pattern = re.compile(r'([{.(-)!_,}])')
    text = specialchar_pattern.sub(" \\1 ", text)
    pattern = r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # Remove unreadable characters  (also extra spaces)
    text = ' '.join(re.sub("[^\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    
    # Optionally, translate text to English
    if detect_translate:
        target_lang='en'
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        text=translated 

    # Optionally, remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        text = [w for w in word_tokens if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, Lemmatize the text
    if lemmatize_text:       
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
        text = " ".join(lemmatized_words)

    return text

@app.get("/")
def hello():
    return {"message":"Hello! Welcome to Ticket Classifier"}

@app.get("/predict-ticketdesc")
def predict_grp(ticketdesc: str):
    """
    A simple function that receives a ticket description and predicts the Group to which this ticket should be assigned.
    :param ticketdesc:
    :return: prediction
    """
    # clean the review
    cleaned_desc = clean_text(ticketdesc)
    
    # perform prediction
    prediction = model.predict([cleaned_desc])
    output = prediction[0]
    
    # show results
    result = {"prediction": output}
    
    return result
