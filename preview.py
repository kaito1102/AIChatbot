import os
import pickle
import sys
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage

from config import OLLAMA_URL, EMBED_MODEL, LLM_MODEL, INDEX_DIR, BM25_PATH, TEMPERATURE, TOP_K

sys.stdout.reconfigure(encoding='utf-8')

DIV = "=" * 55


def fmt_vec(vec, n=20) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in vec[:n]) + ", ...]"


def setup():
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    print(f"✅ System ready  |  embed={EMBED_MODEL}  llm={LLM_MODEL}  top_k={TOP_K}")
    return embeddings, vectorstore, bm25_retriever, llm


def get_chunk_vector(vectorstore, doc) -> np.ndarray | None:
    """Lấy vector gốc của chunk từ FAISS index."""
    try:
        content_hash = hash(doc.page_content)
        inv_map = {v: int(k) for k, v in vectorstore.index_to_docstore_id.items()}
        for doc_id, stored_doc in vectorstore.docstore._dict.items():
            if hash(stored_doc.page_content) == content_hash and doc_id in inv_map:
                return np.array(vectorstore.index.reconstruct(inv_map[doc_id]))
    except Exception:
        pass
    return None


def run_pipeline(query: str, embeddings, vectorstore, bm25_retriever, llm):

    # ── STEP 1: Embed query ───────────────────────────────────
    query_vector = embeddings.embed_query(query)
    q_arr = np.array(query_vector)
    print(f"\n[1] QUERY EMBEDDING")
    print(f"    dim={len(query_vector)}  norm={np.linalg.norm(q_arr):.4f}  "
          f"min={q_arr.min():.4f}  max={q_arr.max():.4f}")
    print(f"    vector (20 dims): {fmt_vec(query_vector)}")

    # ── STEP 2: FAISS search ──────────────────────────────────
    # Lưu lại faiss_score theo doc để dùng ở Step 5
    candidate_k    = TOP_K * 2
    vector_results = vectorstore.similarity_search_with_score(query, k=candidate_k)
    faiss_score_map: dict[int, float] = {}   # hash(page_content) → faiss_score

    print(f"\n[2] FAISS  ({len(vector_results)} hits)")
    for rank, (doc, dist) in enumerate(vector_results, 1):
        score = 1.0 - dist          # cos_sim thực sự theo FAISS
        faiss_score_map[hash(doc.page_content)] = score
        print(f"    #{rank}  cos_sim={score:.4f}  dist={dist:.4f}  "
              f"{doc.metadata.get('filename','?')} p{doc.metadata.get('page','?')}")

    # ── STEP 3: BM25 search ───────────────────────────────────
    bm25_docs = bm25_retriever.invoke(query)[:candidate_k]

    print(f"\n[3] BM25   ({len(bm25_docs)} hits)")
    for rank, doc in enumerate(bm25_docs, 1):
        print(f"    #{rank}  {doc.metadata.get('filename','?')} p{doc.metadata.get('page','?')}")

    # ── STEP 4: RRF fusion ────────────────────────────────────
    rrf_k = 60
    doc_scores: dict = {}
    doc_map:    dict = {}

    def doc_id(doc):
        return (doc.metadata.get("source", "?"), doc.metadata.get("page", 0), hash(doc.page_content))

    for rank, (doc, _) in enumerate(vector_results):
        did = doc_id(doc)
        doc_map[did] = doc
        doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        did = doc_id(doc)
        doc_map[did] = doc
        doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)

    print(f"\n[4] RRF RANKING")
    for i, did in enumerate(sorted_ids, 1):
        doc    = doc_map[did]
        marker = " ← selected" if i <= TOP_K else ""
        print(f"    #{i}  rrf={doc_scores[did]:.5f}  "
              f"{doc.metadata.get('filename','?')} p{doc.metadata.get('page','?')}{marker}")

    # ── STEP 5: Chunk vectors từ FAISS index ──────────────────
    final_ids = sorted_ids[:TOP_K]

    print(f"\n[5] CHUNK VECTORS from FAISS index (Top {len(final_ids)} selected)")

    context_parts = []
    for i, did in enumerate(final_ids, 1):
        doc  = doc_map[did]
        src  = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")

        # Lấy cos_sim từ FAISS score đã tính ở Step 2
        cos_sim = faiss_score_map.get(hash(doc.page_content), float("nan"))
        dist    = 1.0 - cos_sim

        # Lấy vector gốc từ index
        chunk_vec = get_chunk_vector(vectorstore, doc)
        vec_str   = fmt_vec(chunk_vec.tolist()) if chunk_vec is not None else "N/A"

        print(f"    [{i}] {src} p{page}")
        print(f"         cos_sim={cos_sim:.4f}  dist={dist:.4f} ")
        print(f"         vector (20 dims): {vec_str}")

        context_parts.append(f"[Source: {src}, Page {page}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    # ── STEP 6: Call LLM ─────────────────────────────────────
    system_prompt = (
        "You are an intelligent assistant that answers questions strictly based on the provided context. "
        "Answer in detail and accurately. "
        "If the answer is not found in the context, say: 'I could not find this information in the documents.' "
        "Do not fabricate information. Always reply in Vietnamese."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    print(f"\n[6] CALLING LLM  context={len(context)} chars ...")
    response = llm.invoke([HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")])

    print(f"\n{DIV}")
    print(response.content)
    print(f"\n--- Sources ---")
    for i, did in enumerate(final_ids, 1):
        doc   = doc_map[did]
        flags = ("📊" if doc.metadata.get("has_table") else "") + \
                ("🖼️"  if doc.metadata.get("has_image") else "")
        print(f"  [{i}] {flags} {doc.metadata.get('filename','?')}  "
              f"p{doc.metadata.get('page','?')}  rrf={doc_scores[did]:.5f}")
    print(DIV)


def main():
    print(DIV)
    print("  RAG PIPELINE — DEBUG MODE")
    print(DIV)

    embeddings, vectorstore, bm25_retriever, llm = setup()

    while True:
        try:
            query = input("\n💬 Question (or 'exit'): ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Bye!")
                break
            run_pipeline(query, embeddings, vectorstore, bm25_retriever, llm)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()