# 🎭 SarcasmLens

A full NLP pipeline for sarcasm detection in news headlines — combining classical machine learning with hand-engineered linguistic features, wrapped in an interactive, explainable web app.

**🔗 Live demo:** [sarcasmlens-dlpzumxy58lym9uomwa2h6.streamlit.app](https://sarcasmlens-dlpzumxy58lym9uomwa2h6.streamlit.app/)

---

## Overview

SarcasmLens takes any text input, runs it through a six-stage NLP pipeline, and returns:
- A sarcastic / genuine verdict with a confidence score
- Which linguistic features triggered the detection
- Word-level highlights showing each word's contribution to the prediction
- Agreement across four different models (Naive Bayes, Logistic Regression, SVM, Voting Ensemble)

Built as an NLP mini-project, trained on the [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) (Misra & Arora, 2023) — 28,619 headlines sourced from The Onion (sarcastic) and HuffPost (genuine).

---

## The Pipeline

| Stage | What it does |
|---|---|
| **1. Text Preprocessing** | Lowercasing, URL/mention removal, contraction expansion, repeated-character normalization |
| **2. Tokenization & Linguistic Processing** | Word tokenization, POS tagging, stopword removal, lemmatization (NLTK + spaCy) |
| **3. Feature Extraction** | Sentiment (VADER + incongruity score), lexical (TF-IDF, intensifiers, trigger phrases), pragmatic (punctuation, capitalization), structural (POS densities, contrast score) |
| **4. Model Training** | Naive Bayes, Logistic Regression, SVM, and a soft-voting ensemble |
| **5. Feature Importance Analysis** | Logistic Regression coefficients ranked to surface real linguistic insights |
| **6. Deployment** | Interactive Streamlit app with word-level explainability |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 78.0% | 78.0% | 75.1% | 76.5% |
| Logistic Regression | **80.6%** | 80.0% | 79.0% | 79.5% |
| SVM | 79.9% | 80.0% | 77.1% | 78.5% |
| **Voting Ensemble** | **80.5%** | 80.4% | 78.3% | 79.3% |

The Voting Ensemble is used in production — it demonstrably improves robustness over any single model. For example, on the input *"New iPhone Model Set To Launch Next Month"*, standalone Logistic Regression predicts sarcastic (incorrect), while the ensemble correctly predicts genuine, with 3 of 4 models overruling the single wrong vote.

---

## Known Limitations

Because sarcastic headlines come exclusively from The Onion and genuine ones exclusively from HuffPost, the model partially learns **source vocabulary and style**, not sarcasm in a fully general sense. A representative failure case: *"Study Finds Regular Exercise Improves Heart Health"* — a plainly genuine headline — gets misclassified, because "study" is a strong Onion-satire trigger word. This is a well-documented property of this dataset, consistent with findings in published research using the same corpus.

Two real bugs were identified and fixed during development:
1. **Raw noun/verb counts** were dominating predictions regardless of content — fixed by converting to per-token densities.
2. **Stopwords in TF-IDF** ("the", "is", "and") were acting as a proxy for source detection rather than sarcasm — fixed by excluding stopwords and increasing regularization.

A DistilBERT fine-tuning approach was evaluated as an extension (see `src/train_bert.py`) but was not completed — CPU-only training time was not feasible within the project timeline. This is a documented scope decision; published literature suggests fine-tuned transformers could reach ~85-90%+ accuracy on similar tasks.

---

## Project Structure

```
sarcasmlens/
├── data/
│   └── Sarcasm_Headlines_Dataset_v2.json   # raw dataset
├── notebooks/
│   └── eda_and_training.ipynb              # exploratory data analysis
├── src/
│   ├── preprocess.py                       # Stage 1 & 2
│   ├── features.py                         # Stage 3
│   ├── train.py                            # Stage 4
│   ├── evaluate.py                         # Stage 5
│   ├── predict.py                          # inference logic
│   └── train_bert.py                       # DistilBERT experiment (incomplete, see Limitations)
├── models/                                 # trained model artifacts (.pkl)
├── app.py                                  # Streamlit app (Stage 6)
├── requirements.txt
├── runtime.txt
└── setup.sh
```

---

## Running Locally

```bash
git clone https://github.com/shubham-03patil/sarcasmlens.git
cd sarcasmlens
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app.py
```

To retrain from scratch:
```bash
python src/features.py
python src/train.py
python src/evaluate.py
```

---

## Tech Stack

NLTK · spaCy · VADER · Scikit-learn · Streamlit · Streamlit Cloud

---

## Author

Shubham Patil
