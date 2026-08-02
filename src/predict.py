import os
import pickle
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text
from features import (
    get_pragmatic_features, get_sentiment_features,
    get_lexical_features, get_structural_features
)

ENGINEERED_FEATURES = [
    'exclamation_count', 'question_count', 'ellipsis_count', 'caps_ratio', 'quote_count',
    'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'sentiment_incongruity',
    'intensifier_count', 'trigger_phrase_count', 'absolute_word_count',
    'adj_density', 'noun_density', 'verb_density', 'contrast_score'
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def load_all_models():
    with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb") as f:
        nb_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "voting_model.pkl"), "rb") as f:
        voting_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return lr_model, svm_model, nb_model, voting_model, vectorizer, scaler


def extract_features_for_text(raw_text, vectorizer, scaler):
    clean = clean_text(raw_text)

    pragmatic = get_pragmatic_features(raw_text)
    sentiment = get_sentiment_features(clean)
    lexical = get_lexical_features(clean)
    structural = get_structural_features(clean)

    engineered_values = []
    all_feats = {**pragmatic, **sentiment, **lexical, **structural}
    for feat_name in ENGINEERED_FEATURES:
        engineered_values.append(all_feats[feat_name])

    engineered_scaled = scaler.transform([engineered_values])
    tfidf_vec = vectorizer.transform([clean])

    from scipy.sparse import hstack, csr_matrix
    combined = hstack([tfidf_vec, csr_matrix(engineered_scaled)])

    return combined, clean, tfidf_vec


def get_word_contributions(clean_text_str, tfidf_vec, vectorizer, lr_model, top_k=10):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_vec.toarray()[0]
    coefficients = lr_model.coef_[0]

    contributions = []
    nonzero_indices = tfidf_array.nonzero()[0]

    for idx in nonzero_indices:
        word = feature_names[idx]
        weight = tfidf_array[idx]
        coef = coefficients[idx]
        contribution = weight * coef
        contributions.append({'word': word, 'contribution': contribution})

    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return contributions[:top_k]


def predict_sarcasm(raw_text, lr_model, svm_model, nb_model, voting_model, vectorizer, scaler):
    combined_features, clean, tfidf_vec = extract_features_for_text(raw_text, vectorizer, scaler)

    # Voting ensemble is our primary/best model
    voting_pred = voting_model.predict(combined_features)[0]
    voting_proba = voting_model.predict_proba(combined_features)[0]
    confidence = voting_proba[1] if voting_pred == 1 else voting_proba[0]

    lr_pred = lr_model.predict(combined_features)[0]
    svm_pred = svm_model.predict(combined_features)[0]
    nb_pred = nb_model.predict(combined_features)[0]

    word_contributions = get_word_contributions(clean, tfidf_vec, vectorizer, lr_model)

    return {
        'is_sarcastic': bool(voting_pred),
        'confidence': float(confidence),
        'clean_text': clean,
        'word_contributions': word_contributions,
        'model_agreement': {
            'voting_ensemble': bool(voting_pred),
            'logistic_regression': bool(lr_pred),
            'svm': bool(svm_pred),
            'naive_bayes': bool(nb_pred),
        }
    }


if __name__ == "__main__":
    print("Loading models...")
    lr_model, svm_model, nb_model, voting_model, vectorizer, scaler = load_all_models()

    test_headlines = [
        "Man Wins Lottery, Immediately Loses Will To Live",
        "Local Scientists Discover New Species of Frog in Amazon",
        "Area Man Passionate Defender Of What He Imagines Constitution To Be",
        "Study Finds Regular Exercise Improves Heart Health",
        "New iPhone Model Set To Launch Next Month",
    ]

    for headline in test_headlines:
        print(f"\n{'='*60}")
        print(f"Input: {headline}")
        result = predict_sarcasm(headline, lr_model, svm_model, nb_model, voting_model, vectorizer, scaler)
        print(f"Sarcastic: {result['is_sarcastic']} | Confidence: {result['confidence']:.2%}")
        print(f"Model agreement: {result['model_agreement']}")
        print("Top word contributions:")
        for wc in result['word_contributions']:
            direction = "→ sarcastic" if wc['contribution'] > 0 else "→ not sarcastic"
            print(f"   {wc['word']:20s} {wc['contribution']:+.4f} {direction}")