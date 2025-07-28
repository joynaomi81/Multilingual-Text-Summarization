import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader

# Set API keys (Use Streamlit secrets in production)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Set page config
st.set_page_config(page_title="Research Buddy", page_icon="📚")

st.title("📚 Research Assistant Bot")

# Sidebar Menu
menu = st.sidebar.selectbox("Select Feature", [
    "Research Assistant", 
    "Paper Summarizer", 
    "Research Chat Assistant"
])

# 1. Research Assistant
if menu == "Research Assistant":
    st.subheader("📌 Ask Your Research Question")
    field = st.text_input("Enter your academic field (e.g. Linguistics, Computer Science)")
    question = st.text_area("Enter your research question")
    
    if st.button("Get Research Insights"):
        with st.spinner("Thinking..."):
            template = """
            You are a research assistant for the field: {field}. Help answer the following question: {question}.
            Provide:
            - A summary
            - APA style references (if applicable)
            - Relevant academic papers
            - Key researchers in the area
            - Suggested follow-up questions or gaps
            """
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(field=field, question=question)
            st.markdown("### 🔍 Answer:")
            st.write(response)

# 2. Paper Summarizer
elif menu == "Paper Summarizer":
    st.subheader("📄 Summarize Your Paper")
    option = st.radio("Choose Input Method", ["Upload PDF", "Paste Text"])

    if option == "Upload PDF":
        pdf_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if pdf_file is not None:
            reader = PdfReader(pdf_file)
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text()
            st.success("PDF extracted successfully.")
    else:
        raw_text = st.text_area("Paste the research text")

    if st.button("Summarize Paper") and raw_text:
        with st.spinner("Summarizing..."):
            template = """
            Summarize the following research paper content. Highlight:
            - Abstract
            - Main findings
            - Methodology
            - Conclusion
            - 3 key takeaways
            """
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=llm, prompt=prompt)
            summary = chain.run(raw_text)
            st.markdown("### 📝 Summary:")
            st.write(summary)

# 3. Research Chat Assistant
elif menu == "Research Chat Assistant":
    st.subheader("💬 Ask Anything about Research")
    user_input = st.text_area("Ask your question or describe your research interest:")
    if st.button("Get Guidance"):
        with st.spinner("Getting insights..."):
            template = """
            You are a research chatbot. Help this user who wrote:
            "{user_input}"
            Respond with:
            - Suggestions for papers to read
            - Research direction or clarification
            - If unclear, ask follow-up questions
            """
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=llm, prompt=prompt)
            chat_reply = chain.run(user_input=user_input)
            st.markdown("### 🤖 Assistant Reply:")
            st.write(chat_reply)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Joy Olusanya | [GitHub](https://github.com/joynaomi81)")
