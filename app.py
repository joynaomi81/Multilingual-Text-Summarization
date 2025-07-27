import streamlit as st
from transformers import pipeline

# Load QA pipeline
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = load_model()

# Streamlit UI
st.title("🧠 BERT Question Answering App")
st.write("Ask a question based on the context below, and BERT will answer!")

# User input
context = st.text_area("Context Paragraph", height=200, placeholder="Enter your context here...")
question = st.text_input("Question", placeholder="What do you want to ask about the paragraph?")

if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please provide both context and question.")
    else:
        with st.spinner("Thinking..."):
            result = qa_pipeline(question=question, context=context)
            st.success(f"Answer: **{result['answer']}**")
