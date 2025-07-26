#App.py
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

st.title("Multilingual Text Summarizer")
input_text = st.text_area("Enter text in any language", height=200)
lang = st.selectbox("Select input language", ["English", "French", "Yoruba", "German", "Arabic"])

lang_code_map = {
    "English": "en_XX",
    "French": "fr_XX",
    "Yoruba": "yo_XX",
    "German": "de_DE",
    "Arabic": "ar_AR"
}

if st.button("Summarize"):
    tokenizer.src_lang = lang_code_map[lang]
    encoded_input = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # Summarize in English
        max_length=150,
        num_beams=4,
    )
    summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    st.subheader("📝 Summary")
    st.write(summary)
