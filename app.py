from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_ollama import OllamaLLM
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# ---------------------
# OpenAI Model (ChatGPT)
# ---------------------
model = ChatOpenAI(
    model="gpt-3.5-turbo",   # Or gpt-4o-mini, gpt-4.1, etc.
    temperature=0.3
)

# ---------------------
# Ollama Model (Local LLAMA2)
# ---------------------
llm = OllamaLLM(model="llama2")   # Must match the model installed via ollama pull

# ---------------------
# PROMPTS
# ---------------------
prompt1 = ChatPromptTemplate.from_template(
    "Write an essay about {topic} in 25 words."
)

prompt2 = ChatPromptTemplate.from_template(
    "Write a short poem about {topic} suitable for a 5-year-old child in 25 words."
)

# ---------------------
# API Routes
# ---------------------
add_routes(app, prompt1 | model, path="/essay")
add_routes(app, prompt2 | llm, path="/poem")

# ---------------------
# Run server
# ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
