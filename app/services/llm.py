# app/services/llm.py
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"

# 1. Export the official LangChain ChatOllama instance
# This object natively supports .bind_tools() and interfaces perfectly with LangGraph
llm_model = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=8192  # Expanded context length window to comfortably handle OCR + text inputs
)

# 2. Keep your legacy function wrapper for plain prompt scripts
def llm(prompt: str) -> str:
    try:
        response = llm_model.invoke(prompt)
        return response.content
    except Exception as e:
        print("❌ Ollama Error:", e)
        return ""