import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
punct = string.punctuation
stopwords = list(STOP_WORDS)

class CustomTokenizer():
    def __init__(self):
        pass

    def text_data_cleaning(self,sentence):
        doc = nlp(sentence)                         

        tokens = []
        for t in doc:
            if t.lemma_ != "-PRON-":
                temp = t.lemma_.lower().strip()
            else:
              temp = token.lower_
            tokens.append(temp)

        cleaned_tokens = []
        for token in tokens:
            if token not in stopwords and token not in punct:
                cleaned_tokens.append(token)
        return cleaned_tokens