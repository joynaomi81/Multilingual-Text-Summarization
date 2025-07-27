import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
@st.cache_resource
def load_model():
    model_path = "my_finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🌍 Multilingual Text Summarization App")
st.write("This app summarizes text in **any language** using your fine-tuned model.")

# User Input
text_input = st.text_area("✍️ Enter text to summarize", height=250)

# Summarize Button
if st.button("Summarize"):
    if text_input:
        inputs = tokenizer.encode(text_input, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = model.generate(inputs, max_length=200, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("📝 Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")

st.markdown("---")
st.caption("Built with 💙 using Hugging Face Transformers & Streamlit")
