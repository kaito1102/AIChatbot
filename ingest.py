import os
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from smart_pdf_loader import load_directory_smart

# --- Configuration ---
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = os.path.join(BASE_DIR, "data_sources")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
BM25_PATH = os.path.join(BASE_DIR, "bm25_index", "bm25_retriever.pkl")

OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:v1.5"


def build_index():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}'. Add documents and run again.")
        return

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)

    IMAGE_DIR = os.path.join(BASE_DIR, "extracted_images")

    # 1. Load documents — mỗi trang PDF = 1 chunk, KHÔNG split thêm
    print(f"\n📂 Loading documents from {DATA_DIR}...")
    chunks = load_directory_smart(
        DATA_DIR,
        pdf_kwargs={
            "chunk_by":              "page",
            "table_format":          "markdown",
            "image_dir":             IMAGE_DIR,
            "min_image_size":        100,
            "skip_repeated_header":  True,
        },
    )

    if not chunks:
        print("❌ No documents found.")
        return

    table_chunks = sum(1 for c in chunks if c.metadata.get("has_table"))
    image_chunks = sum(1 for c in chunks if c.metadata.get("has_image"))
    print(f"✅ {len(chunks)} page-chunks  ({table_chunks} có bảng, {image_chunks} có ảnh)")
    print("ℹ️  Không dùng RecursiveCharacterTextSplitter — mỗi trang PDF là 1 chunk hoàn chỉnh.")

    # 2. FAISS
    print(f"\n🔢 Generating embeddings ({EMBED_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    print("🚀 Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ FAISS saved → {INDEX_DIR}/")

    # 3. BM25
    print("📝 Building BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25  saved → {BM25_PATH}")

    print("\n🎉 All indexes built successfully!")


if __name__ == "__main__":
    build_index()