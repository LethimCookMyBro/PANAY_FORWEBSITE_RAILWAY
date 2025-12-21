import os
from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="llama3.2",
        base_url="http://ollama:11434",
        temperature=0.2,
    )
