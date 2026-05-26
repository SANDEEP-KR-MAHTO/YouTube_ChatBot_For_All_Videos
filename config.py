from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
STORE_DIR = Path("vectorstores")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4
SCORE_THRESHOLD = 0.0

# ── Models ────────────────────────────────────────────────────────────────────
# Multilingual embedding model — supports 50+ languages including Hindi, free & local
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Multilingual cross-encoder reranker — trained on mMARCO (multilingual MS MARCO)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",   # best for Hindi
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# ── Whisper fallback (Groq API — free tier) ───────────────────────────────────
# Used when no YouTube captions are available for a video.
# Audio is downloaded via yt-dlp, then sent to Groq's whisper-large-v3-turbo.
# Requires: GROQ_API_KEY (same key used for the LLM) + yt-dlp + ffmpeg
# File size limit: 25 MB (~30 min of typical YouTube audio)
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# ── Language preferences ──────────────────────────────────────────────────────
# Language codes tried in order when fetching YouTube captions.
# The user can override this from the sidebar.
PREFERRED_LANGUAGES = ["hi", "en"]

# Map of display name → ISO-639-1 code (None = auto-detect)
LANGUAGE_OPTIONS: dict[str, str | None] = {
    "Auto-detect": None,
    "Hindi (हिन्दी)": "hi",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese": "zh",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
}

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 48

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
