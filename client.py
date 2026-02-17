import requests
import streamlit as st

# -------------------------
# OpenAI Essay Endpoint
# -------------------------
def get_openai_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/essay/invoke",
            json={"input": {"topic": input_text}}
        )
        return response.json()["output"]["content"]
    except:
        return "Server error: Unable to generate essay."

# -------------------------
# Ollama Poem Endpoint
# -------------------------
def get_ollama_response(input_text):
    try:
        response = requests.post(
            "http://localhost:8000/poem/invoke",
            json={"input": {"topic": input_text}}
        )
        return response.json()["output"]
    except:
        return "Server error: Unable to generate poem."

# -------------------------
# Streamlit UI
# -------------------------
st.title("LangChain Demo With LLAMA2 + OpenAI API")

st.subheader("✍️ Generate Essay using OpenAI")
input_text = st.text_input("Write an essay on:")

st.subheader("📝 Generate Poem using LLAMA2 (Ollama)")
input_text1 = st.text_input("Write a poem on:")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))
