import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import build_tfidf_features

ENGINEERED_FEATURES = [
    'exclamation_count', 'question_count', 'ellipsis_count', 'caps_ratio', 'quote_count',
    'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'sentiment_incongruity',
    'intensifier_count', 'trigger_phrase_count', 'absolute_word_count',
    'adj_density', 'noun_density', 'verb_density', 'contrast_score'
]


def load_features_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "..", "data", "features_dataset.csv")
    df = pd.read_csv(path)
    df['clean_headline'] = df['clean_headline'].fillna('')
    return df


def build_combined_features(df):
    vectorizer, tfidf_matrix = build_tfidf_features(df['clean_headline'], max_features=3000)
    engineered = df[ENGINEERED_FEATURES].values
    scaler = MinMaxScaler()
    engineered_scaled = scaler.fit_transform(engineered)
    engineered_sparse = csr_matrix(engineered_scaled)
    combined = hstack([tfidf_matrix, engineered_sparse])
    return combined, vectorizer, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test):
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42, C=2.0, class_weight='balanced')
    svm_base = LinearSVC(random_state=42, max_iter=5000, class_weight='balanced')
    svm = CalibratedClassifierCV(svm_base, cv=3)  # wraps SVM so it can output probabilities

    models = {
        'naive_bayes': nb,
        'logistic_regression': lr,
        'svm': svm,
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        trained_models[name] = model

        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # ---- Voting Classifier: combines all 3 models' predictions ----
    print("\nTraining voting_ensemble...")
    voting_clf = VotingClassifier(
        estimators=[('nb', nb), ('lr', lr), ('svm', svm)],
        voting='soft'  # uses predicted probabilities, not just majority vote
    )
    voting_clf.fit(X_train, y_train)
    voting_preds = voting_clf.predict(X_test)

    acc = accuracy_score(y_test, voting_preds)
    prec = precision_score(y_test, voting_preds)
    rec = recall_score(y_test, voting_preds)
    f1 = f1_score(y_test, voting_preds)

    results['voting_ensemble'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    trained_models['voting_ensemble'] = voting_clf

    print(f"voting_ensemble -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return trained_models, results


def save_artifacts(trained_models, vectorizer, scaler):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")

    with open(os.path.join(models_dir, "nb_model.pkl"), "wb") as f:
        pickle.dump(trained_models['naive_bayes'], f)
    with open(os.path.join(models_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(trained_models['logistic_regression'], f)
    with open(os.path.join(models_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(trained_models['svm'], f)
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "voting_model.pkl"), "wb") as f:
        pickle.dump(trained_models['voting_ensemble'], f)
    print(f"\nAll models saved to {models_dir}")


if __name__ == "__main__":
    print("Loading features dataset...")
    df = load_features_dataset()

    print("Building combined feature matrix (TF-IDF + engineered features)...")
    X, vectorizer, scaler = build_combined_features(df)
    y = df['is_sarcastic'].values

    print("Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_models, results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n=== FINAL COMPARISON ===")
    results_df = pd.DataFrame(results).T
    print(results_df)

    save_artifacts(trained_models, vectorizer, scaler)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_df.to_csv(os.path.join(script_dir, "..", "data", "model_results.csv"))
    print("Results saved to data/model_results.csv")