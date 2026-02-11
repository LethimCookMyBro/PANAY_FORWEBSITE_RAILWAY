# backend/app/utils.py
"""
Shared resource getters for FastAPI app.
These resources are initialized in main.py lifespan and stored in app.state.
"""
import os
from langchain_ollama import OllamaLLM

# Module-level singleton (set by main.py during startup)
_llm_instance = None


def set_llm(llm):
    """Called by main.py lifespan to share the LLM instance"""
    global _llm_instance
    _llm_instance = llm


def get_llm():
    """Get the shared LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        # Fallback: create one if not initialized (shouldn't happen in normal flow)
        _llm_instance = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            num_predict=int(os.getenv("LLM_NUM_PREDICT", "1024")),
        )
    return _llm_instance
