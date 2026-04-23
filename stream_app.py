import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_model()

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Write a movie review in English 👇")

review = st.text_area(
    "✍️ Enter your review here:",
    placeholder="This movie was absolutely amazing...",
    height=150,
)

if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=200, padding="post")
            prob = model.predict(padded)[0][0]

        st.divider()
        if prob > 0.5:
            st.success("### ✨ Positive")
            st.progress(float(prob))
            st.metric("Confidence Score", f"{prob*100:.1f}%")
        else:
            st.error("### 👎 Negative")
            st.progress(float(1 - prob))
            st.metric("Confidence Score", f"{(1-prob)*100:.1f}%")

st.divider()
st.caption("Sentiment Analysis — ANN vs LSTM vs BERT")
