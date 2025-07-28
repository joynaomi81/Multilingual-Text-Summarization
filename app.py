import os
import streamlit as st
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults

# 🔐 API keys (hide with secrets in deployment)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# 🔧 Initialize LLM and tools
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)
search_tool = TavilySearchResults()
tools = [search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# 💾 Save response to local file
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
st.markdown("Ask research questions and get real-time responses with academic resources and APA citations.")

field = st.text_input("Enter your research field (e.g. Linguistics, AI, Medicine):")
question = st.text_area("What do you want to research?", height=150)

if st.button("🔍 Search"):
    if field and question:
        full_prompt = (
            f"You are a helpful research assistant for a graduate student in the field of {field}. "
            f"The student wants to research: '{question}'.\n"
            f"Give a summary of the topic using real-time search. Include links to academic journals or papers, and mention any relevant researchers. "
            f"Also include citations in APA format if available."
        )

        with st.spinner("Searching..."):
            try:
                response = agent.run(full_prompt)
                st.success("✅ Response generated!")
                st.markdown("### 📄 Result:")
                st.write(response)

                if st.checkbox("💾 Save this response?"):
                    save_response(field, question, response)
                    st.success("Saved to local file.")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
    else:
        st.warning("Please fill in both the field and your question.")
