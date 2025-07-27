import streamlit as st
from transformers import pipeline

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt")
    qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
    return summarizer, qa_pipeline

summarizer, qa_pipeline = load_models()

st.title("🌍 Multilingual Summarization & Question Answering")

# Input section
option = st.selectbox("Choose Task", ["Summarization", "Question Answering"])

if option == "Summarization":
    text = st.text_area("Enter text to summarize:", height=300)
    if st.button("Summarize"):
        if text.strip():
            with st.spinner("Summarizing..."):
                summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.success("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

elif option == "Question Answering":
    context = st.text_area("Enter context passage:", height=300)
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if context.strip() and question.strip():
            with st.spinner("Answering..."):
                result = qa_pipeline(question=question, context=context)
            st.success("Answer:")
            st.write(result['answer'])
        else:
            st.warning("Please enter both context and question.")
