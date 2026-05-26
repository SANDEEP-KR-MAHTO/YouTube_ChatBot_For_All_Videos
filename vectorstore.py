import json
import logging
import pickle
import shutil
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import (
    CACHE_TTL_HOURS,
    EMBEDDING_MODEL,
    SCORE_THRESHOLD,
    STORE_DIR,
    TOP_K_RETRIEVE,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

def _store_path(video_id: str) -> Path:
    return STORE_DIR / video_id


# ── Embeddings (cached singleton) ─────────────────────────────────────────────

_embeddings = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},  # enables cosine similarity
        )
    return _embeddings


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _is_cache_valid(store_path: Path) -> bool:
    meta_file = store_path / "meta.json"
    if not meta_file.exists():
        return store_path.exists()  # old cache without meta — treat as valid
    with open(meta_file) as f:
        meta = json.load(f)
    age_hours = (time.time() - meta.get("created_at", 0)) / 3600
    if age_hours >= CACHE_TTL_HOURS:
        return False
    # Invalidate if the embedding model has changed (e.g. switched to multilingual)
    cached_model = meta.get("embedding_model")
    if cached_model and cached_model != EMBEDDING_MODEL:
        logger.info(
            f"Cache built with '{cached_model}', current model is '{EMBEDDING_MODEL}' — rebuilding."
        )
        return False
    return True


def get_cache_info(video_id: str) -> dict | None:
    """Return cache metadata dict for a video, or None if not cached."""
    meta_file = _store_path(video_id) / "meta.json"
    if not meta_file.exists():
        return None
    with open(meta_file) as f:
        return json.load(f)


def clear_cache(video_id: str | None = None) -> None:
    """Delete the vectorstore cache for one video or all videos."""
    if video_id:
        path = _store_path(video_id)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Cleared cache for {video_id}")
    else:
        if STORE_DIR.exists():
            shutil.rmtree(STORE_DIR)
            logger.info("Cleared all vectorstore caches")


# ── Build ─────────────────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: list[dict], video_id: str
) -> tuple[FAISS, BM25Okapi, list[Document]]:
    """
    Build a FAISS vectorstore + BM25 index from timestamp-aware chunks.

    Args:
        chunks   – list of {"text", "start", "end"} dicts from transcript.py
        video_id – YouTube video ID

    Returns:
        (vectorstore, bm25_index, docs_list)
    """
    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "video_id": video_id,
                "chunk_index": i,
                "start": chunk["start"],
                "end": chunk["end"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    logger.info(f"Building FAISS vectorstore for {video_id} with {len(docs)} chunks")
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # BM25 index over the same docs
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    # Persist to disk
    store_path = _store_path(video_id)
    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_path))

    with open(store_path / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, docs), f)

    with open(store_path / "meta.json", "w") as f:
        json.dump(
            {
                "created_at": time.time(),
                "video_id": video_id,
                "chunk_count": len(docs),
                "embedding_model": EMBEDDING_MODEL,
            },
            f,
        )

    logger.info(f"Saved vectorstore to {store_path}")
    return vectorstore, bm25, docs


# ── Load ──────────────────────────────────────────────────────────────────────

def load_vectorstore(
    video_id: str,
) -> tuple[FAISS | None, BM25Okapi | None, list[Document] | None]:
    store_path = _store_path(video_id)
    if not store_path.exists() or not _is_cache_valid(store_path):
        return None, None, None

    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(
            str(store_path), embeddings, allow_dangerous_deserialization=True
        )

        bm25_path = store_path / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                bm25, docs = pickle.load(f)
        else:
            bm25, docs = None, None

        logger.info(f"Loaded cached vectorstore for {video_id}")
        return vectorstore, bm25, docs
    except Exception as e:
        logger.warning(f"Failed to load cached vectorstore for {video_id}: {e}")
        return None, None, None


def get_or_build_vectorstore(
    chunks: list[dict], video_id: str
) -> tuple[FAISS, BM25Okapi, list[Document]]:
    vectorstore, bm25, docs = load_vectorstore(video_id)
    if vectorstore is not None:
        return vectorstore, bm25, docs
    return build_vectorstore(chunks, video_id)


# ── Hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(
    vectorstore: FAISS,
    bm25: BM25Okapi | None,
    docs_corpus: list[Document] | None,
    query: str,
    k: int = TOP_K_RETRIEVE,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[Document]:
    """
    Reciprocal Rank Fusion of FAISS (semantic) + BM25 (keyword) results.

    With normalized embeddings, FAISS L2 distance maps to cosine similarity as:
        cosine_sim = 1 - dist / 2   (range 0–1)
    """
    # ── FAISS semantic search ─────────────────────────────────────────────────
    faiss_results = vectorstore.similarity_search_with_score(query, k=k)
    faiss_ranked: list[Document] = []
    for doc, dist in faiss_results:
        cosine_sim = max(0.0, 1.0 - dist / 2.0)
        if cosine_sim >= score_threshold:
            faiss_ranked.append(doc)

    # ── BM25 keyword search ───────────────────────────────────────────────────
    bm25_ranked: list[Document] = []
    if bm25 is not None and docs_corpus:
        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_ranked = [docs_corpus[i] for i in top_indices if scores[i] > 0]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    RRF_K = 60
    rrf: dict[int, float] = {}

    for rank, doc in enumerate(faiss_ranked):
        key = doc.metadata.get("chunk_index", id(doc))
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(bm25_ranked):
        key = doc.metadata.get("chunk_index", id(doc))
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Collect unique docs sorted by RRF score
    seen: set[int] = set()
    merged: list[Document] = []
    for doc in faiss_ranked + bm25_ranked:
        key = doc.metadata.get("chunk_index", id(doc))
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    merged.sort(
        key=lambda d: rrf.get(d.metadata.get("chunk_index", id(d)), 0.0),
        reverse=True,
    )

    logger.debug(
        f"Hybrid search → {len(merged)} unique docs "
        f"(FAISS: {len(faiss_ranked)}, BM25: {len(bm25_ranked)})"
    )
    return merged[:k]
