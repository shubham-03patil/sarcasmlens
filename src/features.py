import re
import numpy as np
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

INTENSIFIERS = {'very', 'extremely', 'totally', 'absolutely', 'literally',
                 'completely', 'utterly', 'so', 'really', 'incredibly'}

SARCASM_TRIGGERS = {'yeah right', 'oh great', 'oh good', 'oh joy', 'wow',
                     'shocking', 'surprise surprise', 'what a surprise',
                     'totally', 'obviously', 'clearly', 'genius'}


def get_pragmatic_features(text):
    exclamation_count = text.count('!')
    question_count = text.count('?')
    ellipsis_count = text.count('...') + len(re.findall(r'\.\.(?!\.)', text))
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    quote_count = text.count('"') + text.count("'")
    return {
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'ellipsis_count': ellipsis_count,
        'caps_ratio': caps_ratio,
        'quote_count': quote_count,
    }


def get_sentiment_features(clean_text):
    scores = vader.polarity_scores(clean_text)
    words = clean_text.split()
    if len(words) >= 4:
        mid = len(words) // 2
        first_half = ' '.join(words[:mid])
        second_half = ' '.join(words[mid:])
        first_score = vader.polarity_scores(first_half)['compound']
        second_score = vader.polarity_scores(second_half)['compound']
        incongruity = abs(first_score - second_score)
    else:
        incongruity = 0.0
    return {
        'vader_neg': scores['neg'],
        'vader_neu': scores['neu'],
        'vader_pos': scores['pos'],
        'vader_compound': scores['compound'],
        'sentiment_incongruity': incongruity,
    }


def get_lexical_features(clean_text):
    words = clean_text.split()
    intensifier_count = sum(1 for w in words if w in INTENSIFIERS)
    trigger_count = sum(1 for phrase in SARCASM_TRIGGERS if phrase in clean_text)
    return {
        'intensifier_count': intensifier_count,
        'trigger_phrase_count': trigger_count,
    }


def get_structural_features(clean_text):
    doc = nlp(clean_text)
    total_tokens = len(doc) if len(doc) > 0 else 1
    adj_count = sum(1 for token in doc if token.pos_ == 'ADJ')
    adj_density = adj_count / total_tokens
    noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
    verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
    noun_density = noun_count / total_tokens
    verb_density = verb_count / total_tokens
    contrast_words = {'but', 'however', 'yet', 'although', 'though', 'despite'}
    contrast_score = sum(1 for token in doc if token.text in contrast_words)
    return {
        'adj_density': adj_density,
        'noun_density': noun_density,
        'verb_density': verb_density,
        'contrast_score': contrast_score,
    }


def extract_all_features(df):
    pragmatic_list = []
    sentiment_list = []
    lexical_list = []
    structural_list = []

    for i, row in df.iterrows():
        raw = row['headline']
        clean = row['clean_headline']
        pragmatic_list.append(get_pragmatic_features(raw))
        sentiment_list.append(get_sentiment_features(clean))
        lexical_list.append(get_lexical_features(clean))
        structural_list.append(get_structural_features(clean))
        if i % 5000 == 0:
            print(f"Processed {i}/{len(df)} rows...")

    pragmatic_df = pd.DataFrame(pragmatic_list)
    sentiment_df = pd.DataFrame(sentiment_list)
    lexical_df = pd.DataFrame(lexical_list)
    structural_df = pd.DataFrame(structural_list)

    features_df = pd.concat(
        [df.reset_index(drop=True), pragmatic_df, sentiment_df, lexical_df, structural_df],
        axis=1
    )
    return features_df


def build_tfidf_features(clean_headlines, max_features=3000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=3
    )
    tfidf_matrix = vectorizer.fit_transform(clean_headlines)
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_clean_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "Sarcasm_Headlines_Dataset_v2.json")

    print("Loading and cleaning dataset...")
    df = load_and_clean_dataset(data_path)

    print("Extracting linguistic features (this takes a few minutes)...")
    features_df = extract_all_features(df)

    print("\nFeature columns added:")
    print(features_df.columns.tolist())
    print("\nSample rows:")
    print(features_df[['headline', 'vader_compound', 'sentiment_incongruity',
                        'exclamation_count', 'caps_ratio', 'adj_density']].head())

    out_path = os.path.join(script_dir, "..", "data", "features_dataset.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")