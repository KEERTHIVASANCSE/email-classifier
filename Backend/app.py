
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"

st.set_page_config(
    page_title="SpamShield | Email Classifier",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🛡️ SpamShield")
st.caption("Machine learning powered email spam detection")

if not MODEL_PATH.exists():
    st.error("Model not found. Run python train.py first.")
    st.stop()

model = load_model()

email = st.text_area(
    "Paste your email",
    height=220,
    placeholder="Paste the subject and email body here...",
)

if st.button("Analyze Email", type="primary", use_container_width=True):
    if not email.strip():
        st.warning("Please enter some email text.")
    else:
        prediction = int(model.predict([email])[0])
        probabilities = model.predict_proba([email])[0]
        spam_index = list(model.classes_).index(1)
        spam_probability = probabilities[spam_index]

        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("🚨 This email is classified as SPAM.")
        else:
            st.success("✅ This email is classified as NOT SPAM.")

        st.metric(
            "Estimated spam probability",
            f"{spam_probability:.2%}",
        )

        st.progress(float(spam_probability))

        st.caption(
            "Predictions are estimates and may be incorrect."
        )
