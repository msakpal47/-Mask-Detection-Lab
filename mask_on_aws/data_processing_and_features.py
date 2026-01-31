import re
import numpy as np

def text_data_cleaning(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def tfidf_features_transform(cleaned_text: str, vectorizer):
    # vectorizer is a fitted sklearn.feature_extraction.text.TfidfVectorizer
    vec = vectorizer.transform([cleaned_text])
    return vec
