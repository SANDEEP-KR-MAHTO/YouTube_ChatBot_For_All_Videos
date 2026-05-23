import logging
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from config import (
    DEFAULT_GROQ_MODEL,
    GROQ_MODELS,
    LANGUAGE_OPTIONS,
    LOG_FORMAT,
    LOG_LEVEL,
    WHISPER_MODEL_SIZE,
)
from rag_chain import (
    format_docs_with_timestamps,
    generate_summary,
    retrieve_and_rerank,
    stream_response,
)
from transcript import format_timestamp, get_transcript_chunks
from vectorstore import clear_cache, get_cache_info, get_or_build_vectorstore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 YouTube Video Chatbot")
st.caption("Paste a YouTube URL, then ask anything about the video.")

# ── Session state init ────────────────────────────────────────────────────────
# videos: {video_id: {vectorstore, bm25, docs, url, summary, chat_history}}
if "videos" not in st.session_state:
    st.session_state.videos = {}
if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def active_video() -> dict | None:
    vid = st.session_state.active_video_id
    return st.session_state.videos.get(vid)


def export_chat_markdown(url: str, history: list[dict]) -> str:
    lines = [
        f"# YouTube Video Chatbot — Exported Chat",
        f"**Video URL:** {url}",
        f"**Exported at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for turn in history:
        role = "**You**" if turn["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {turn['content']}")
        lines.append("")
    return "\n".join(lines)


def render_sources(source_docs) -> None:
    """Render an expandable source-highlighting section below an answer."""
    if not source_docs:
        return
    with st.expander(f"📄 Sources used ({len(source_docs)} excerpts)", expanded=False):
        for i, doc in enumerate(source_docs, 1):
            start = doc.metadata.get("start")
            end = doc.metadata.get("end")
            if start is not None and end is not None:
                ts = f"`{format_timestamp(start)} – {format_timestamp(end)}`"
            else:
                ts = ""
            st.markdown(f"**Excerpt {i}** {ts}")
            st.markdown(f"> {doc.page_content}")
            if i < len(source_docs):
                st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # LLM selector
    selected_model = st.selectbox(
        "LLM Model (Groq — all free)",
        options=GROQ_MODELS,
        index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL),
        help="All models run on Groq's free tier.",
    )

    st.divider()

    # Language selector
    selected_lang_label = st.selectbox(
        "Video Language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Select the spoken language of the video. Used to pick the right YouTube caption track and guide Whisper.",
    )
    selected_lang_code: str | None = LANGUAGE_OPTIONS[selected_lang_label]

    # Whisper model size selector
    whisper_size = st.selectbox(
        "Whisper Model (fallback)",
        options=["tiny", "base", "small", "medium", "large"],
        index=["tiny", "base", "small", "medium", "large"].index(WHISPER_MODEL_SIZE),
        help=(
            "Used only when no YouTube captions are found.\n"
            "tiny/base = fast | small = good for Hindi | medium/large = best quality (needs more RAM)"
        ),
    )

    st.divider()

    # Model info
    st.markdown("**Stack**")
    st.markdown("- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (local)")
    st.markdown("- Reranker: `mmarco-mMiniLMv2-L12` (local, multilingual)")
    st.markdown("- Vector DB: FAISS + BM25 (local)")
    st.markdown(f"- LLM: `{selected_model}` (Groq)")

    st.divider()

    # Loaded videos — multi-video switcher
    st.markdown("**Loaded Videos**")
    if st.session_state.videos:
        video_options = {
            vid: data["url"] for vid, data in st.session_state.videos.items()
        }
        chosen = st.radio(
            "Switch video",
            options=list(video_options.keys()),
            format_func=lambda v: video_options[v][:50] + "…"
            if len(video_options[v]) > 50
            else video_options[v],
            index=list(video_options.keys()).index(st.session_state.active_video_id)
            if st.session_state.active_video_id in video_options
            else 0,
        )
        if chosen != st.session_state.active_video_id:
            st.session_state.active_video_id = chosen
            st.rerun()
    else:
        st.caption("No videos loaded yet.")

    st.divider()

    # Cache management
    st.markdown("**Cache Management**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Current", use_container_width=True):
            vid = st.session_state.active_video_id
            if vid:
                clear_cache(vid)
                del st.session_state.videos[vid]
                st.session_state.active_video_id = (
                    next(iter(st.session_state.videos), None)
                )
                st.success("Cache cleared.")
                st.rerun()
    with col2:
        if st.button("Clear All", use_container_width=True):
            clear_cache()
            st.session_state.videos = {}
            st.session_state.active_video_id = None
            st.success("All caches cleared.")
            st.rerun()

    av = active_video()
    if av:
        info = get_cache_info(st.session_state.active_video_id)
        if info:
            from datetime import datetime as _dt
            ts = _dt.fromtimestamp(info["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Index built: {ts} · {info.get('chunk_count', '?')} chunks")


# ── Video loader ──────────────────────────────────────────────────────────────
with st.form("video_form"):
    url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    load_btn = st.form_submit_button("Load Video", type="primary")

if load_btn and url:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
    else:
        with st.spinner("Fetching transcript and building index…"):
            try:
                chunks, video_id, transcript_source = get_transcript_chunks(
                    url,
                    language=selected_lang_code,
                    whisper_model_size=whisper_size,
                )

                # Reset chat only for brand-new videos
                existing = st.session_state.videos.get(video_id, {})
                chat_history = existing.get("chat_history", [])

                vectorstore, bm25, docs = get_or_build_vectorstore(chunks, video_id)

                # Auto-generate summary (reuse cached if same video reloaded)
                if existing.get("summary"):
                    summary = existing["summary"]
                else:
                    with st.spinner("Generating video summary…"):
                        summary = generate_summary(docs, model=selected_model)

                word_count = sum(len(c["text"].split()) for c in chunks)
                st.session_state.videos[video_id] = {
                    "vectorstore": vectorstore,
                    "bm25": bm25,
                    "docs": docs,
                    "url": url,
                    "summary": summary,
                    "chat_history": chat_history,
                    "word_count": word_count,
                    "transcript_source": transcript_source,
                    "language": selected_lang_label,
                }
                st.session_state.active_video_id = video_id

                source_badge = (
                    "📺 YouTube captions"
                    if transcript_source == "youtube"
                    else f"🎙️ Whisper ({whisper_size})"
                )
                st.success(
                    f"✅ Video loaded! "
                    f"{word_count:,} words · {len(docs)} chunks · {source_badge}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Failed to load video")

# ── Main chat area ────────────────────────────────────────────────────────────
av = active_video()

if av:
    # Video summary
    src = av.get("transcript_source", "youtube")
    lang = av.get("language", "")
    src_label = (
        "📺 YouTube captions"
        if src == "youtube"
        else f"🎙️ Whisper transcription"
    )
    with st.expander(f"📋 Video Summary  ·  {src_label}  ·  {lang}", expanded=True):
        st.markdown(av["summary"])

    st.divider()

    # Chat history
    for i, turn in enumerate(av["chat_history"]):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            # Show sources for assistant turns that have them
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])

    # Chat input
    user_query = st.chat_input("Ask anything about the video…")

    if user_query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found.")
        else:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    # Retrieve + rerank sources first (fast)
                    source_docs = retrieve_and_rerank(
                        av["vectorstore"], av["bm25"], av["docs"], user_query
                    )
                    context = format_docs_with_timestamps(source_docs)

                    # Stream the LLM response
                    response = st.write_stream(
                        stream_response(
                            context,
                            user_query,
                            av["chat_history"],
                            model=selected_model,
                        )
                    )

                    # Show sources inline below the answer
                    render_sources(source_docs)

                    # Persist to chat history
                    av["chat_history"].append(
                        {"role": "user", "content": user_query}
                    )
                    av["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": source_docs,
                        }
                    )

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logger.exception("Generation failed")

    # ── Chat controls ─────────────────────────────────────────────────────────
    if av["chat_history"]:
        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                av["chat_history"] = []
                st.rerun()
        with ctrl_col2:
            md = export_chat_markdown(av["url"], av["chat_history"])
            st.download_button(
                label="⬇️ Export Chat",
                data=md,
                file_name="chat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )

else:
    st.info("⬆️ Paste a YouTube URL above and click **Load Video** to start chatting.")
