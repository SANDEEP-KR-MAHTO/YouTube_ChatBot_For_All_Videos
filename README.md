# YouTube Video Chatbot (RAG)

A multilingual conversational AI chatbot that lets you ask questions about any YouTube video — including Hindi videos. Built with **Retrieval-Augmented Generation (RAG)**. Completely free to run, no paid APIs required.

---

## How It Works

```
YouTube URL
    ↓
Try fetching YouTube captions (language-aware)
    ↓ (if no captions available)
Download audio (yt-dlp) → Transcribe locally (OpenAI Whisper)
    ↓
Timestamp-aware chunking
    ↓
Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    ↓
Hybrid search: FAISS (semantic) + BM25 (keyword) → Reciprocal Rank Fusion
    ↓
Cross-encoder reranking (mmarco-mMiniLMv2)
    ↓
Streamed answer via Groq LLaMA / Gemma (free)
```

---

## Features

### Multilingual Support
- Works with **Hindi, English, and 10+ other languages**
- Language-aware YouTube caption fetching — tries your selected language first
- **Whisper fallback** — if no YouTube captions exist, audio is downloaded and transcribed locally for free using OpenAI Whisper
- Multilingual embedding and reranking models handle non-English content correctly

### Retrieval Quality
- **Hybrid search** — combines FAISS semantic search with BM25 keyword search using Reciprocal Rank Fusion for better coverage
- **Cross-encoder reranking** — re-scores retrieved chunks with a multilingual cross-encoder before sending to the LLM
- **Timestamp-aware chunking** — transcript is split into chunks that preserve the original video timestamps
- **Score-based filtering** — low-relevance chunks are filtered out before reranking

### User Interface
- **Streaming responses** — answers appear token by token as they are generated
- **Timestamp citations** — each source excerpt shows the video timestamp range (e.g. `1:24 – 2:10`)
- **Source highlighting** — expandable panel below every answer shows exactly which transcript excerpts were used
- **Auto video summary** — a 4–6 bullet summary of the whole video is generated on load
- **Multi-video support** — load multiple videos and switch between them in the sidebar without losing chat history
- **LLM selector** — choose from 5 free Groq models in the sidebar
- **Export chat** — download the full conversation as a Markdown file
- **Cache management** — clear the index for one video or all videos from the sidebar; shows when the index was built

### Performance & Reliability
- **Disk caching with TTL** — vectorstores are saved to disk and reused across sessions; automatically rebuilt after 48 hours or if the embedding model changes
- **Singleton model loading** — embedding model, reranker, and Whisper model are each loaded once and reused
- **Structured logging** — retrieval scores, chunk counts, and model info are logged for debugging

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| UI | Streamlit | Free |
| YouTube captions | youtube-transcript-api | Free |
| Audio download (fallback) | yt-dlp | Free |
| Audio transcription (fallback) | OpenAI Whisper (`small`) | Free, runs locally |
| ffmpeg binaries | imageio-ffmpeg | Free, auto-downloaded |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Free, runs locally |
| Vector store | FAISS | Free, runs locally |
| Keyword search | BM25 (rank-bm25) | Free, runs locally |
| Reranker | `mmarco-mMiniLMv2-L12-H384-v1` | Free, runs locally |
| LLM | Groq (LLaMA 3.1 / Gemma / Mixtral) | Free tier |

---

## Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> First run will download the embedding model (~420 MB), reranker (~120 MB), and Whisper model (~460 MB for `small`). These are one-time downloads cached locally.

**4. Set up your API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

**5. Run the app**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Usage

1. Open the sidebar and select the **video language** and **LLM model**
2. Paste a YouTube URL and click **Load Video**
   - If the video has captions, they are fetched instantly
   - If not, audio is downloaded and transcribed locally via Whisper (takes a few minutes)
3. Read the auto-generated **video summary**
4. Ask questions in the chat box — answers stream in real time
5. Click **📄 Sources used** below any answer to see the exact transcript excerpts
6. Use the sidebar to **switch between loaded videos** or **clear the cache**
7. Click **⬇️ Export Chat** to download the conversation as Markdown

---

## Project Structure

```
RAG/
├── app.py              # Streamlit UI — all interface logic
├── rag_chain.py        # LCEL chain, hybrid retrieval, reranking, streaming, summary
├── vectorstore.py      # FAISS + BM25 build/load, hybrid search, cache management
├── transcript.py       # YouTube caption fetch + Whisper fallback transcription
├── config.py           # Central config — models, chunk sizes, thresholds, languages
├── requirements.txt    # Python dependencies
├── .env                # Your API keys (never commit this)
├── .env.example        # Template for .env
└── vectorstores/       # Cached FAISS indexes (auto-created, one folder per video)
```

---

## Configuration

All tunable parameters live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding model |
| `RERANKER_MODEL` | `mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder reranker |
| `WHISPER_MODEL_SIZE` | `small` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |
| `DEFAULT_GROQ_MODEL` | `llama-3.1-8b-instant` | Default LLM |
| `CHUNK_SIZE` | `800` | Max characters per transcript chunk |
| `TOP_K_RETRIEVE` | `10` | Chunks fetched by hybrid search |
| `TOP_K_RERANK` | `4` | Chunks kept after reranking |
| `CACHE_TTL_HOURS` | `48` | Hours before a cached index is rebuilt |

**Whisper model size guide:**

| Size | Download | Speed | Hindi quality |
|---|---|---|---|
| `tiny` | ~75 MB | Very fast | Basic |
| `base` | ~140 MB | Fast | Decent |
| `small` | ~460 MB | Moderate | Good |
| `medium` | ~1.5 GB | Slow | Very good |
| `large` | ~3 GB | Slowest | Best |

---

## Supported Languages

YouTube caption languages and Whisper transcription both support these and more:

Hindi, English, Spanish, French, German, Japanese, Chinese, Arabic, Portuguese, Russian

Select the language in the sidebar before loading a video.

---

## Limitations

- **Whisper transcription is slow** for long videos — a 1-hour video may take 5–15 minutes on CPU depending on the model size
- **LLM answers in the transcript's language** — if the video is in Hindi and you ask in English, the LLM may respond in either language depending on the model
- **Private or age-restricted videos** cannot be accessed by either the caption API or yt-dlp

---

## License

MIT License — free to use and modify.
