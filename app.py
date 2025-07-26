# app.py
import streamlit as st
from transformers import pipeline
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"  # Covers 45+ languages
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

st.title("🌍 Multilingual Text Summarization App")
st.markdown("This app summarizes text in **multiple languages** using mT5.")

text_input = st.text_area("Enter your text here", height=300)

if text_input:
    with st.spinner("Detecting language and summarizing..."):
        lang = detect(text_input)
        summarizer = load_model()
        summary = summarizer(text_input, max_length=128, min_length=30, do_sample=False)
        st.success(f"Detected Language: `{lang}`")
        st.subheader("📝 Summary:")
        st.write(summary[0]['summary_text'])
