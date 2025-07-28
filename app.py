import os
import streamlit as st
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from PyPDF2 import PdfReader

# ✅ Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# ✅ Initialize LLM and search tool
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
tavily = TavilySearchResults()

# ✅ Streamlit config
st.set_page_config(page_title="Research Buddy", page_icon="📚")
st.title("📚 Research Assistant Bot")

# ✅ Sidebar Menu
menu = st.sidebar.selectbox("Select Feature", [
    "Research Assistant", 
    "Paper Summarizer", 
    "Research Chat Assistant"
])

# ✅ Search real-time papers with Tavily
def fetch_research_links(topic):
    try:
        results = tavily.run(topic)
        links = "\n".join([f"- [{res['title']}]({res['url']})" for res in results[:5]])
        return links
    except Exception as e:
        return f"⚠ Could not fetch live results: {e}"

# 🔎 1. Research Assistant
if menu == "Research Assistant":
    st.subheader("📌 Ask Your Research Question")
    field = st.text_input("Enter your academic field (e.g. Linguistics, Computer Science)")
    question = st.text_area("Enter your research question")

    if st.button("Get Research Insights"):
        if field and question:
            with st.spinner("Thinking and researching..."):
                links = fetch_research_links(question)

                template = """
                You are a research assistant in the field of {field}. A user asked: "{question}".
                Help by providing:
                - A brief summary
                - Why this topic matters
                - Gaps or follow-up directions
                - APA references if possible
                - Mention relevant authors or institutions
                """
                prompt = PromptTemplate.from_template(template)
                chain = LLMChain(llm=llm, prompt=prompt)
                reasoning = chain.run(field=field, question=question)

                st.markdown("### 🤖 Assistant Insights:")
                st.write(reasoning)

                st.markdown("### 🌐 Real-Time Papers / Resources:")
                st.markdown(links, unsafe_allow_html=True)
        else:
            st.warning("Please fill in both your field and research question.")

# 📄 2. Paper Summarizer
elif menu == "Paper Summarizer":
    st.subheader("📄 Summarize Your Paper")
    option = st.radio("Choose Input Method", ["Upload PDF", "Paste Text"])
    raw_text = ""

    if option == "Upload PDF":
        pdf_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if pdf_file is not None:
            reader = PdfReader(pdf_file)
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
            - Key findings
            - Methodology
            - Conclusion
            - 3 takeaways
            """
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=llm, prompt=prompt)
            summary = chain.run(raw_text)
            st.markdown("### 📝 Summary:")
            st.write(summary)

# 💬 3. Research Chat Assistant
elif menu == "Research Chat Assistant":
    st.subheader("💬 Ask Anything about Research")
    user_input = st.text_area("Ask a question or describe your topic of interest:")
    
    if st.button("Get Guidance"):
        if user_input:
            with st.spinner("Analyzing and researching..."):
                links = fetch_research_links(user_input)

                template = """
                You are a research chatbot. A user asked: "{user_input}".
                Help them by:
                - Suggesting what to read
                - Recommending paper titles or keywords
                - Offering brief clarification or direction
                - Mentioning scholars or institutions if relevant
                - Include reasons or follow-up suggestions
                """
                prompt = PromptTemplate.from_template(template)
                chain = LLMChain(llm=llm, prompt=prompt)
                reply = chain.run(user_input=user_input)

                st.markdown("### 🤖 Assistant Reply:")
                st.write(reply)

                st.markdown("### 🌐 Suggested Readings:")
                st.markdown(links, unsafe_allow_html=True)
        else:
            st.warning("Please describe your topic.")

# 🔚 Footer
st.markdown("---")
st.markdown("Built with ❤ by Joy Olusanya | [GitHub](https://github.com/joynaomi81)")
