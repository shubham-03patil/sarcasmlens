import re
import contractions

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Expand contractions (don't -> do not)
    text = contractions.fix(text)
    
    # Normalize repeated characters (soooo -> soo, max 2 repeats)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    sample = "OMGGGG I can't believe @user posted this https://theonion.com/link !!! sooooo funny"
    print(clean_text(sample))
import re
import contractions
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = contractions.fix(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_process(text):
    """
    Takes cleaned text, returns:
    - tokens: list of words (no stopwords, lemmatized)
    - pos_tags: list of (word, POS tag) pairs
    """
    doc = nlp(text)
    
    tokens = []
    pos_tags = []
    
    for token in doc:
        pos_tags.append((token.text, token.pos_))
        # Skip stopwords and punctuation, keep meaningful words
        if token.text not in stop_words and not token.is_punct and token.text.strip() != '':
            lemma = lemmatizer.lemmatize(token.text)
            tokens.append(lemma)
    
    return tokens, pos_tags


def load_and_clean_dataset(filepath):
    """Load dataset, apply cleaning, return dataframe"""
    df = pd.read_json(filepath, lines=True)
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df


if __name__ == "__main__":
    sample = "OMGGGG I can't believe @user posted this https://theonion.com/link !!! sooooo funny"
    cleaned = clean_text(sample)
    print("Cleaned:", cleaned)
    
    tokens, pos_tags = tokenize_and_process(cleaned)
    print("Tokens:", tokens)
    print("POS Tags:", pos_tags)
    
    # Test on full dataset - path works regardless of where script is run from
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "Sarcasm_Headlines_Dataset_v2.json")
    
    df = load_and_clean_dataset(data_path)
    print("\nDataset shape:", df.shape)
    print(df[['headline', 'clean_headline']].head())