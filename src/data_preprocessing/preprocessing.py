import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class DataPreprocessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Remove usernames, timestamps, and non-textual data
        text = re.sub(r'\[.*?\]', '', text)  # Remove [something]
        text = re.sub(r'\(.*?\)', '', text)  # Remove (something)
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)  # Remove timestamps
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        return text.strip()
    
    def normalize_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess_conversation(self, conversation):
        cleaned = self.clean_text(conversation)
        normalized = self.normalize_text(cleaned)
        return normalized