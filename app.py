import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import math

# Load model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = load_model()

# Function to chunk text
def chunk_text(text, max_tokens=400, stride=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - stride
    return chunks

# Streamlit UI
st.title("📚 BERT QA on PDF (Chunk-Aware)")
st.write("Upload a PDF and ask a question. The app splits the PDF into chunks to improve answer accuracy.")

# Upload and extract PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
pdf_text = ""

if uploaded_file:
    with st.spinner("Extracting text..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()
        st.success("Text extracted successfully!")

    with st.expander("🔍 Preview Extracted Text"):
        st.write(pdf_text[:3000] + "..." if len(pdf_text) > 3000 else pdf_text)

# Ask question
question = st.text_input("❓ Ask your question:")

if st.button("Get Answer"):
    if not pdf_text or not question:
        st.warning("Please upload a PDF and enter your question.")
    else:
        with st.spinner("Thinking..."):
            chunks = chunk_text(pdf_text)
            best_answer = None
            best_score = 0

            for chunk in chunks:
                try:
                    result = qa_pipeline(question=question, context=chunk)
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_answer = result['answer']
                except:
                    continue

            if best_answer:
                st.success(f"✅ Answer: **{best_answer}** (Confidence: {best_score:.2f})")
            else:
                st.error("Sorry, I couldn’t find a good answer in the document.")
