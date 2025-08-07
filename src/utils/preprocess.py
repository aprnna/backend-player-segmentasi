import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Memuat model Spacy
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Dictionary normalisasi
normalization_dict = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "plz": "please",
    "thx": "thanks", "ty": "thank you", "btw": "by the way", "lol": "laughing out loud",
    "lmao": "laughing my ass off", "omg": "oh my god", "wtf": "what the fuck",
    "idk": "i do not know", "imo": "in my opinion", "imho": "in my humble opinion",
    "afaik": "as far as i know", "brb": "be right back", "bbl": "be back later",
    "gg": "good game", "ez": "easy", "nerf": "weaken", "buff": "strengthen",
    "noob": "newbie", "grind": "repetitive task", "op": "overpowered",
    "laggy": "slow connection", "gitgud": "get good", "f2p": "free to play",
    "p2w": "pay to win", "dlc": "downloadable content", "npc": "non player character",
    "fps": "frames per second", "afk": "away from keyboard", "xp": "experience points",
    "lvl": "level", "bossfight": "boss fight", "rng": "random number generator",
    "camping": "staying in one spot", "gank": "ambush attack", "re": "regarding",
    "af": "as fuck", "xd": "bad", "f": "fuck", "ive": "i have", "rp": "roleplay",
    "fckin": "fucking", "tl": "too long", "dr": "didnt read", "lcg": "living card game", 
    "stori": "story", "differ": "different"
}

def normalize(text):
    return ' '.join([normalization_dict.get(word, word) for word in text.split()])

def lemmatize(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def preprocess_text(text):
    text = normalize(text)
    tokens = text.split()
    tokens = lemmatize(tokens)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)
