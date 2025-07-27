import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import fitz  # PyMuPDF

# === Load models ===
@st.cache_resource
def load_qa_models():
    en_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    multi_model = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
    return en_model, multi_model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

qa_en, qa_multi = load_qa_models()
summarizer = load_summarizer()

# === Helper functions ===
def chunk_text(text, max_tokens=400, stride=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - stride
    return chunks

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

def get_best_answer(question, context_chunks, qa_pipeline):
    best_answer = None
    best_score = 0
    for chunk in context_chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
        except:
            continue
    return best_answer, best_score

# === Streamlit UI ===
st.title("📚🌍 Multilingual BERT QA with Summarization")
st.write("Upload a PDF, get a summary, and ask questions in English or Yoruba!")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
pdf_text = ""

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()

    st.success("✅ Text extracted successfully!")

    with st.expander("📖 Preview Extracted Text"):
        st.write(pdf_text[:3000] + "..." if len(pdf_text) > 3000 else pdf_text)

    # === Language detection ===
    lang = detect_lang(pdf_text)
    st.info(f"🔤 Detected language: **{lang}**")

    # === Summarization ===
    if len(pdf_text.split()) > 50:
        st.subheader("📝 Document Summary")
        summary_output = summarizer(pdf_text[:1024], max_length=150, min_length=40, do_sample=False)
        st.write(summary_output[0]['summary_text'])

    # === Question answering ===
    question = st.text_input("❓ Ask your question:")
    if st.button("Get Answer"):
        with st.spinner("Searching for answers..."):
            chunks = chunk_text(pdf_text)
            qa_model = qa_en if lang == "en" else qa_multi
            answer, score = get_best_answer(question, chunks, qa_model)

            if answer:
                st.success(f"✅ Answer: **{answer}** (Confidence: {score:.2f})")
            else:
                st.warning("No confident answer found in the document.")
