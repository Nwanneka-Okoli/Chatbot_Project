# import the libraries
import nltk
nltk.download('stopwords')

# tokenizes and removes punctuation
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# get the root of the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import json


# load json file using load function
def load_data():
    with open("intents.json") as json_file:
        chatbot_data = json.load(json_file)
        return chatbot_data


# splits the sentences into words
def sentence_tokenize(sentence):
    return tokenizer.tokenize(sentence)


# gets the root of each word passed to it.
def word_stem(w):
    return stemmer.stem(w.lower())
