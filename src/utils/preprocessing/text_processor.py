import re
import nltk
import spacy
from nltk.corpus import stopwords
import ssl
import spacy.cli
import json
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
# locate the JSON next to this script
BASE = os.path.dirname(__file__)
NORMALIZATION_PATH = os.path.join(BASE, "normalization_dict.json")

with open(NORMALIZATION_PATH, 'r', encoding='utf-8') as f:
    normalization_dict = json.load(f)

def preprocess_single_text(text):
    # 1. Case Folding
    text = text.lower()
    # 2. Cleansing (Menghapus non-alphanumeric, angka, dsb)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Normalisasi
    text = ' '.join([normalization_dict.get(word, word) for word in text.split()])
    # 4. Tokenisasi
    tokens = nltk.word_tokenize(text)
    # 5. Lemmatisasi
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    # 6. Stopword Removal
    final_tokens = [word for word in lemmatized_tokens if word not in stop_words]
    return ' '.join(final_tokens)