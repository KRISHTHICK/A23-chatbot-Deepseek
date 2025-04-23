# ===============================
# Required Packages (pip install)
# ===============================
# pip install streamlit
# pip install langchain
# pip install ollama

import streamlit as st
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize LLM using Ollama with DeepSeek or Gemma or any local model
@st.cache_resource
def get_llm(model_name="deepseek-coder"):
    return Ollama(model=model_name)

# Setup conversation chain with memory
def create_conversation_chain(model_name="deepseek-coder"):
    llm = get_llm(model_name)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)

# Streamlit Chat UI
st.set_page_config(page_title="🧠 Local Ollama Chatbot", layout="wide")
st.title("🧠 Local Chatbot with DeepSeek/Gemma/LLaMA (Ollama)")

model_choice = st.sidebar.selectbox("Choose a local Ollama model:", ["deepseek-coder", "gemma", "llama3"])

if "chain" not in st.session_state or st.session_state.get("model_name") != model_choice:
    st.session_state.chain = create_conversation_chain(model_name=model_choice)
    st.session_state.model_name = model_choice
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    response = st.session_state.chain.run(user_input)
    st.session_state.chat_history.append((user_input, response))

if st.session_state.chat_history:
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**{model_choice.capitalize()}:** {a}")
