import streamlit as st
from transformers import pipeline
import PyPDF2

# Load QA pipeline
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = load_model()

# Streamlit UI
st.title("📄🤖 Ask Questions from PDF using BERT")
st.write("Upload a PDF, and ask questions based on its content!")

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

pdf_text = ""
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

    st.success("Text extracted from PDF successfully!")

# Show extracted content (optional preview)
if pdf_text:
    with st.expander("📖 Preview Extracted Text"):
        st.write(pdf_text[:3000])  # Only preview first 3000 characters

# Question answering section
question = st.text_input("❓ Ask your question based on the PDF")

if st.button("Get Answer"):
    if not pdf_text or not question.strip():
        st.warning("Please upload a PDF and enter your question.")
    else:
        with st.spinner("Thinking..."):
            result = qa_pipeline(question=question, context=pdf_text)
            st.success(f"Answer: **{result['answer']}**")
