import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import load_all_models, predict_sarcasm

st.set_page_config(page_title="SarcasmLens", page_icon="🎭", layout="centered")

st.title("🎭 SarcasmLens")
st.caption("A full NLP pipeline for sarcasm detection with linguistic feature analysis")


@st.cache_resource
def get_models():
    return load_all_models()


lr_model, svm_model, nb_model, vectorizer, scaler = get_models()

text_input = st.text_area(
    "Enter a headline or sentence:",
    placeholder="e.g. Area Man Passionate Defender Of What He Imagines Constitution To Be",
    height=100
)

if st.button("Analyze", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict_sarcasm(text_input, lr_model, svm_model, nb_model, vectorizer, scaler)

        # ---- Main verdict ----
        if result['is_sarcastic']:
            st.error(f"### 🎭 SARCASTIC")
        else:
            st.success(f"### ✅ NOT SARCASTIC")

        st.metric("Confidence", f"{result['confidence']:.1%}")

        # ---- Word-level highlights ----
        st.subheader("Word-level Sarcasm Cues")
        st.caption("Green = pushes toward sarcastic | Red = pushes toward not sarcastic")

        highlighted_html = ""
        words_in_order = result['clean_text'].split()
        contribution_map = {wc['word']: wc['contribution'] for wc in result['word_contributions']}

        for word in words_in_order:
            if word in contribution_map:
                contrib = contribution_map[word]
                if contrib > 0:
                    intensity = min(abs(contrib) * 40, 100)
                    color = f"rgba(255, 0, 0, {intensity/100:.2f})"
                else:
                    intensity = min(abs(contrib) * 40, 100)
                    color = f"rgba(0, 150, 255, {intensity/100:.2f})"
                highlighted_html += f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px; margin:1px;">{word}</span> '
            else:
                highlighted_html += f'{word} '

        st.markdown(highlighted_html, unsafe_allow_html=True)

        # ---- Top contributing words table ----
        st.subheader("Top Contributing Words")
        for wc in result['word_contributions'][:8]:
            direction = "→ Sarcastic" if wc['contribution'] > 0 else "→ Not Sarcastic"
            st.write(f"**{wc['word']}**: {wc['contribution']:+.4f} {direction}")

        # ---- Model comparison ----
        st.subheader("Model Agreement")
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression", "Sarcastic" if result['model_agreement']['logistic_regression'] else "Not Sarcastic")
        col2.metric("SVM", "Sarcastic" if result['model_agreement']['svm'] else "Not Sarcastic")
        col3.metric("Naive Bayes", "Sarcastic" if result['model_agreement']['naive_bayes'] else "Not Sarcastic")

st.divider()
st.caption("Built with NLTK, spaCy, VADER, and Scikit-learn | SarcasmLens NLP Project")