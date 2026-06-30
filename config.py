import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # If .env does not exist, try loading default environment variables
    load_dotenv()

# Ollama API settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")

# Helper to resolve paths relative to BASE_DIR if they are relative
def get_path(env_key: str, default_rel_path: str) -> str:
    path_val = os.getenv(env_key)
    if not path_val:
        return str(BASE_DIR / default_rel_path)
    p = Path(path_val)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)

# File and Directory Paths
DATA_DIR = get_path("DATA_DIR", "data_sources")
INDEX_DIR = get_path("INDEX_DIR", "faiss_index")
BM25_PATH = get_path("BM25_PATH", "bm25_index/bm25_retriever.pkl")
IMAGE_DIR = get_path("IMAGE_DIR", "extracted_images")

# Retrieval & Generation settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("TOP_K", "4"))

def print_config():
    """Debug utility to print the loaded configuration."""
    print("==================================================")
    print("          CHATBOT CONFIGURATION SYSTEM            ")
    print("==================================================")
    print(f"Ollama URL:       {OLLAMA_URL}")
    print(f"Embedding Model:  {EMBED_MODEL}")
    print(f"LLM Chat Model:   {LLM_MODEL}")
    print(f"Data Source Dir:  {DATA_DIR}")
    print(f"FAISS Index Dir:  {INDEX_DIR}")
    print(f"BM25 Index Path:  {BM25_PATH}")
    print(f"Extracted Images: {IMAGE_DIR}")
    print(f"Temperature:      {TEMPERATURE}")
    print(f"Top-K Retrieval:  {TOP_K}")
    print("==================================================")

if __name__ == "__main__":
    print_config()
