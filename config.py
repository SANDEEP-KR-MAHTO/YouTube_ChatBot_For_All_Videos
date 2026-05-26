from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STORE_DIR = Path("vectorstores")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4
SCORE_THRESHOLD = 0.0

# ── Embedding model (multilingual, free, local) ───────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Reranker model ────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ── Groq LLM ─────────────────────────────────────────────────────────────────
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 48

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
