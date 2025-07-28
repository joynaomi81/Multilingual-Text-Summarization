import os
import streamlit as st
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults

# 🔐 API keys (secured using Streamlit secrets)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# 🔧 Initialize LLM and tools
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
search_tool = TavilySearchResults()
tools = [search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# 💾 Save response to file
def save_response(field, question, response):
    filename = f"research_output_{field.replace(' ', '_').lower()}.txt"
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        file.write(f"Field: {field}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Response:\n{response}\n")

# 🎨 Streamlit UI
st.set_page_config(page_title="Research Assistant", page_icon="📚")

st.title("📚 AI Research Assistant")
st.markdown("Get answers to research questions or summarize any research text. Powered by Gemini + Tavily + Langchain.")

# 📌 Sidebar menu
menu = st.sidebar.selectbox("📋 Select Task", ["Research Q&A", "Summarizer"])

# ================= RESEARCH Q&A ==================
if menu == "Research Q&A":
    st.subheader("🔍 Ask Research Questions")
    field = st.text_input("Enter your research field (e.g. Linguistics, AI, Medicine):")
    question = st.text_area("What do you want to research?", height=150)

    if st.button("🔎 Search"):
        if field and question:
            prompt = (
                f"You are a helpful research assistant for a graduate student in the field of {field}. "
                f"The student wants to research: '{question}'.\n"
                f"Give a summary of the topic using real-time search. Include links to academic journals or papers, "
                f"mention relevant researchers, and provide citations in APA format."
            )
            with st.spinner("Generating response..."):
                try:
                    response = agent.run(prompt)
                    st.success("✅ Response generated!")
                    st.markdown("### 📄 Result:")
                    st.write(response)

                    if st.checkbox("💾 Save this response?"):
                        save_response(field, question, response)
                        st.success("Saved to local file.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {e}")
        else:
            st.warning("Please enter both your field and question.")

# ================= SUMMARIZER ==================
elif menu == "Summarizer":
    st.subheader("📚 Summarize Research Text")
    text_to_summarize = st.text_area("Paste your research abstract, paragraph, or article:", height=200)

    if st.button("📝 Summarize"):
        if text_to_summarize.strip():
            summarization_prompt = (
                "Summarize the following research text in clear academic language with key points:\n\n"
                f"{text_to_summarize}"
            )
            with st.spinner("Summarizing..."):
                try:
                    summary = llm.invoke(summarization_prompt).content
                    st.markdown("### ✨ Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"⚠️ Summarization failed: {e}")
        else:
            st.warning("Please paste some text to summarize.")
