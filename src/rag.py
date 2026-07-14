import os
import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KNOWLEDGE_DIR, VECTORSTORE_DIR,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
)

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

INDEX_PATH  = os.path.join(VECTORSTORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

_embedder = None
_index    = None
_chunks   = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> list[dict]:
    words   = text.split()
    chunks  = []
    step    = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i: i + CHUNK_SIZE]
        if len(chunk_words) < 20:
            continue
        chunks.append({"text": " ".join(chunk_words), "source": source})
    return chunks


# ── Build vectorstore ──────────────────────────────────────────────────────────

def build_vectorstore(force: bool = False):
    if not force and os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("[RAG] Vectorstore already exists. Skipping build.")
        return

    embedder = _get_embedder()
    all_chunks = []

    print("[RAG] Loading knowledge docs...")
    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(KNOWLEDGE_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_text(text, source=fname.replace(".txt", ""))
        all_chunks.extend(chunks)
        print(f"  {fname}: {len(chunks)} chunks")

    print(f"\n[RAG] Total chunks: {len(all_chunks)}")
    print("[RAG] Encoding chunks...")

    texts      = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine after normalization
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[RAG] Vectorstore saved: {index.ntotal} vectors  dim={dim}")


# ── Load vectorstore ───────────────────────────────────────────────────────────

def _load_vectorstore():
    global _index, _chunks
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Vectorstore not found. Run build_vectorstore() first.")
        _index  = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)


# ── Retrieve ───────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    _load_vectorstore()
    embedder   = _get_embedder()
    q_embed    = embedder.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_embed)
    scores, idxs = _index.search(q_embed, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    return results


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_vectorstore(force=True)

    print("\n[RAG] Test retrieval: 'melanoma ABCDE dermoscopy'")
    results = retrieve("melanoma ABCDE dermoscopy signs")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] source={r['source']}  score={r['score']}")
        print(f"       {r['text'][:200]}...")
