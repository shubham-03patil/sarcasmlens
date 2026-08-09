import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import load_all_models, predict_sarcasm

st.set_page_config(page_title="SarcasmLens — Copy Desk", page_icon="🖋", layout="centered")

# ---------------------------------------------------------------
# THEME: Editorial / Copy Desk — ink stamps, red & blue pen marks
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap');

:root {
    --ink: #15141B;
    --surface: #1E1D26;
    --surface-2: #26242F;
    --border: #38364230;
    --border-solid: #38364280;
    --paper: #E9E6DC;
    --paper-dim: #9B97A6;
    --red-pen: #C1432B;
    --red-pen-dim: #C1432B22;
    --blue-pen: #4F7CA6;
    --blue-pen-dim: #4F7CA622;
    --gold: #C9A15A;
}

.stApp {
    background:
        radial-gradient(circle at 15% -10%, #211f2b 0%, transparent 45%),
        var(--ink);
    color: var(--paper);
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 720px;}

/* ---------- Masthead ---------- */
.masthead {
    text-align: center;
    padding-bottom: 1.4rem;
    margin-bottom: 2rem;
    border-bottom: 3px double var(--border-solid);
}
.masthead .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.28em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.masthead h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 3rem;
    letter-spacing: -0.01em;
    margin: 0;
    color: var(--paper);
}
.masthead .dek {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--paper-dim);
    margin-top: 0.5rem;
    letter-spacing: 0.03em;
}

/* ---------- Input area ---------- */
.slip-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    color: var(--paper) !important;
    border: 1px solid var(--border-solid) !important;
    border-radius: 3px !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 1.05rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: none !important;
}

.stButton > button {
    background: transparent !important;
    color: var(--paper) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    color: var(--ink) !important;
}

/* ---------- Verdict Stamp ---------- */
.stamp-wrap {
    display: flex;
    justify-content: center;
    margin: 2.2rem 0 1.6rem 0;
}
.stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.4rem;
    letter-spacing: 0.12em;
    padding: 0.65rem 1.8rem;
    border: 3px solid currentColor;
    border-radius: 4px;
    transform: rotate(-4deg);
    text-transform: uppercase;
    display: inline-block;
    position: relative;
}
.stamp::before {
    content: "";
    position: absolute;
    inset: 4px;
    border: 1px solid currentColor;
    border-radius: 2px;
    opacity: 0.5;
}
.stamp.sarcastic { color: var(--red-pen); background: var(--red-pen-dim); }
.stamp.genuine { color: var(--blue-pen); background: var(--blue-pen-dim); }

/* ---------- Confidence readout ---------- */
.readout {
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    color: var(--paper-dim);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.readout .value {
    color: var(--paper);
    font-size: 1rem;
    font-weight: 600;
}
.meter {
    height: 4px;
    background: var(--surface-2);
    border-radius: 2px;
    max-width: 360px;
    margin: 0.5rem auto 2rem auto;
    overflow: hidden;
}
.meter-fill { height: 100%; }
.meter-fill.sarcastic { background: var(--red-pen); }
.meter-fill.genuine { background: var(--blue-pen); }

/* ---------- Section labels ---------- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--gold);
    text-transform: uppercase;
    margin: 1.8rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-solid);
}

/* ---------- Annotated manuscript ---------- */
.manuscript {
    font-family: 'Source Serif 4', serif;
    font-size: 1.25rem;
    line-height: 2.1;
    padding: 1.2rem 1.4rem;
    background: var(--surface);
    border: 1px solid var(--border-solid);
    border-radius: 3px;
}
.mark {
    padding-bottom: 2px;
    border-bottom: 2px solid;
}
.mark.pos { border-color: var(--red-pen); }
.mark.neg { border-color: var(--blue-pen); }

/* ---------- Contribution list ---------- */
.contrib-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
}
.contrib-word { color: var(--paper); }
.contrib-val.pos { color: var(--red-pen); }
.contrib-val.neg { color: var(--blue-pen); }

/* ---------- Vote chips ---------- */
.vote-row { display: flex; gap: 0.6rem; margin-top: 0.6rem; flex-wrap: wrap; }
.vote-chip {
    flex: 1;
    min-width: 120px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.7rem 0.5rem;
    border-radius: 3px;
    border: 1px solid;
}
.vote-chip .model-name { display: block; color: var(--paper-dim); margin-bottom: 0.3rem; font-size: 0.62rem; }
.vote-chip.sarcastic { border-color: var(--red-pen); color: var(--red-pen); background: var(--red-pen-dim); }
.vote-chip.genuine { border-color: var(--blue-pen); color: var(--blue-pen); background: var(--blue-pen-dim); }

.footer-note {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--paper-dim);
    letter-spacing: 0.08em;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border-solid);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# MASTHEAD
# ---------------------------------------------------------------
st.markdown("""
<div class="masthead">
    <div class="eyebrow">Est. NLP Copy Desk &middot; Linguistic Forensics Unit</div>
    <h1>SarcasmLens</h1>
    <div class="dek">6-STAGE PIPELINE &middot; TF-IDF + LINGUISTIC FEATURES &middot; ENSEMBLE VERDICT</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_models():
    return load_all_models()


lr_model, svm_model, nb_model, voting_model, vectorizer, scaler = get_models()

st.markdown('<div class="slip-label">Submit text for review</div>', unsafe_allow_html=True)
text_input = st.text_area(
    "input",
    placeholder="e.g. Area Man Passionate Defender Of What He Imagines Constitution To Be",
    height=100,
    label_visibility="collapsed"
)

_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    analyze_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if analyze_clicked:
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = predict_sarcasm(text_input, lr_model, svm_model, nb_model, voting_model, vectorizer, scaler)
        verdict_class = "sarcastic" if result['is_sarcastic'] else "genuine"
        verdict_text = "Sarcastic" if result['is_sarcastic'] else "Genuine"
        conf_pct = result['confidence'] * 100

        # ---- Stamp ----
        st.markdown(f"""
        <div class="stamp-wrap">
            <div class="stamp {verdict_class}">{verdict_text}</div>
        </div>
        <div class="readout">Ensemble Confidence <span class="value">{conf_pct:.1f}%</span></div>
        <div class="meter"><div class="meter-fill {verdict_class}" style="width:{conf_pct:.0f}%"></div></div>
        """, unsafe_allow_html=True)

        # ---- Annotated manuscript ----
        st.markdown('<div class="section-label">Annotated Copy</div>', unsafe_allow_html=True)
        contribution_map = {wc['word']: wc['contribution'] for wc in result['word_contributions']}
        words_in_order = result['clean_text'].split()

        marked_words = []
        for word in words_in_order:
            if word in contribution_map:
                contrib = contribution_map[word]
                mark_class = "pos" if contrib > 0 else "neg"
                marked_words.append(f'<span class="mark {mark_class}">{word}</span>')
            else:
                marked_words.append(word)

        st.markdown(f'<div class="manuscript">{" ".join(marked_words)}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.68rem; color:var(--paper-dim); margin-top:0.5rem;">'
            '<span style="color:var(--red-pen)">— red underline</span> pushes sarcastic &nbsp;&middot;&nbsp; '
            '<span style="color:var(--blue-pen)">— blue underline</span> pushes genuine</div>',
            unsafe_allow_html=True
        )

        # ---- Top contributing words ----
        st.markdown('<div class="section-label">Marginal Notes</div>', unsafe_allow_html=True)
        rows_html = ""
        for wc in result['word_contributions'][:8]:
            cls = "pos" if wc['contribution'] > 0 else "neg"
            rows_html += f"""
            <div class="contrib-row">
                <span class="contrib-word">{wc['word']}</span>
                <span class="contrib-val {cls}">{wc['contribution']:+.3f}</span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)

        # ---- Model agreement ----
        st.markdown('<div class="section-label">Editorial Board Vote</div>', unsafe_allow_html=True)
        agreement = result['model_agreement']
        labels = {
            'voting_ensemble': 'Ensemble',
            'logistic_regression': 'Log. Reg.',
            'svm': 'SVM',
            'naive_bayes': 'Naive Bayes'
        }
        chips_html = '<div class="vote-row">'
        for key, label in labels.items():
            cls = "sarcastic" if agreement[key] else "genuine"
            verdict = "Sarcastic" if agreement[key] else "Genuine"
            chips_html += f"""
            <div class="vote-chip {cls}">
                <span class="model-name">{label}</span>{verdict}
            </div>"""
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
    NLTK &middot; spaCy &middot; VADER &middot; Scikit-learn &middot; SarcasmLens NLP Pipeline
</div>
""", unsafe_allow_html=True)