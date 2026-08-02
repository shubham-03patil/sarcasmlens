import os
import pickle
import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    'exclamation_count', 'question_count', 'ellipsis_count', 'caps_ratio', 'quote_count',
    'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'sentiment_incongruity',
    'intensifier_count', 'trigger_phrase_count', 'absolute_word_count',
    'adj_density', 'noun_density', 'verb_density', 'contrast_score'
]

def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    with open(os.path.join(models_dir, "lr_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return lr_model, vectorizer


def get_feature_importance(lr_model, vectorizer, top_n=20):
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    all_feature_names = tfidf_feature_names + ENGINEERED_FEATURES
    coefficients = lr_model.coef_[0]

    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': coefficients
    })
    importance_df['feature_type'] = importance_df['feature'].apply(
        lambda f: 'engineered' if f in ENGINEERED_FEATURES else 'word (tfidf)'
    )

    top_sarcastic = importance_df.sort_values('coefficient', ascending=False).head(top_n)
    top_not_sarcastic = importance_df.sort_values('coefficient', ascending=True).head(top_n)

    return top_sarcastic, top_not_sarcastic, importance_df


def print_linguistic_insights(top_sarcastic, top_not_sarcastic):
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES PUSHING TOWARD 'SARCASTIC'")
    print("=" * 60)
    for _, row in top_sarcastic.iterrows():
        print(f"  {row['feature']:30s} | coef: {row['coefficient']:.4f} | type: {row['feature_type']}")

    print("\n" + "=" * 60)
    print("TOP 20 FEATURES PUSHING TOWARD 'NOT SARCASTIC'")
    print("=" * 60)
    for _, row in top_not_sarcastic.iterrows():
        print(f"  {row['feature']:30s} | coef: {row['coefficient']:.4f} | type: {row['feature_type']}")

    engineered_in_top = top_sarcastic[top_sarcastic['feature_type'] == 'engineered']
    print("\n" + "=" * 60)
    print("ENGINEERED LINGUISTIC FEATURES IN TOP SARCASTIC SIGNALS")
    print("=" * 60)
    if len(engineered_in_top) > 0:
        print(engineered_in_top[['feature', 'coefficient']].to_string(index=False))
    else:
        print("  None in top 20 — word patterns (TF-IDF) dominate over engineered features here.")


if __name__ == "__main__":
    print("Loading trained Logistic Regression model...")
    lr_model, vectorizer = load_models()

    print("Computing feature importance...")
    top_sarcastic, top_not_sarcastic, full_importance_df = get_feature_importance(lr_model, vectorizer, top_n=20)

    print_linguistic_insights(top_sarcastic, top_not_sarcastic)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "..", "data", "feature_importance.csv")
    full_importance_df.to_csv(out_path, index=False)
    print(f"\nFull feature importance table saved to {out_path}")