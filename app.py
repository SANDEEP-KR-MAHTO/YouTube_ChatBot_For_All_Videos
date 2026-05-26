import logging
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_GROQ_MODEL, GROQ_MODELS, LOG_FORMAT, LOG_LEVEL
from rag_chain import (
    format_docs_with_timestamps,
    generate_summary,
    retrieve_and_rerank,
    stream_response,
)
from transcript import NoCaptionsError, format_timestamp, get_transcript_chunks
from vectorstore import clear_cache, get_cache_info, get_or_build_vectorstore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# On Streamlit Cloud secrets live in st.secrets — copy into os.environ
for _key in ["GROQ_API_KEY"]:
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except (KeyError, FileNotFoundError):
            pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 YouTube Video Chatbot")
st.caption("Paste a YouTube URL and ask anything about the video.")

# ── Session state ─────────────────────────────────────────────────────────────
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
        "# YouTube Video Chatbot — Exported Chat",
        f"**Video URL:** {url}",
        f"**Exported at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "", "---", "",
    ]
    for turn in history:
        role = "**You**" if turn["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {turn['content']}")
        lines.append("")
    return "\n".join(lines)


def render_sources(source_docs) -> None:
    if not source_docs:
        return
    with st.expander(f"📄 Sources ({len(source_docs)} excerpts)", expanded=False):
        for i, doc in enumerate(source_docs, 1):
            start = doc.metadata.get("start")
            end = doc.metadata.get("end")
            ts = (
                f"`{format_timestamp(start)} – {format_timestamp(end)}`"
                if start is not None and end is not None
                else ""
            )
            st.markdown(f"**Excerpt {i}** {ts}")
            st.markdown(f"> {doc.page_content}")
            if i < len(source_docs):
                st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    selected_model = st.selectbox(
        "LLM Model",
        options=GROQ_MODELS,
        index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL),
        help="All models are free via Groq.",
    )

    st.divider()
    st.markdown("**Stack**")
    st.markdown("- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`")
    st.markdown("- Reranker: `mmarco-mMiniLMv2-L12`")
    st.markdown("- Vector DB: FAISS + BM25")
    st.markdown(f"- LLM: `{selected_model}` (Groq free)")

    st.divider()

    # Multi-video switcher
    st.markdown("**Loaded Videos**")
    if st.session_state.videos:
        video_options = {
            vid: data["url"] for vid, data in st.session_state.videos.items()
        }
        chosen = st.radio(
            "Switch video",
            options=list(video_options.keys()),
            format_func=lambda v: (
                video_options[v][:50] + "…"
                if len(video_options[v]) > 50
                else video_options[v]
            ),
            index=(
                list(video_options.keys()).index(st.session_state.active_video_id)
                if st.session_state.active_video_id in video_options
                else 0
            ),
        )
        if chosen != st.session_state.active_video_id:
            st.session_state.active_video_id = chosen
            st.rerun()
    else:
        st.caption("No videos loaded yet.")

    st.divider()

    # Cache management
    st.markdown("**Cache**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Current", use_container_width=True):
            vid = st.session_state.active_video_id
            if vid:
                clear_cache(vid)
                del st.session_state.videos[vid]
                st.session_state.active_video_id = next(
                    iter(st.session_state.videos), None
                )
                st.success("Cleared.")
                st.rerun()
    with col2:
        if st.button("Clear All", use_container_width=True):
            clear_cache()
            st.session_state.videos = {}
            st.session_state.active_video_id = None
            st.success("Cleared.")
            st.rerun()

    av = active_video()
    if av:
        info = get_cache_info(st.session_state.active_video_id)
        if info:
            from datetime import datetime as _dt
            ts = _dt.fromtimestamp(info["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Index: {ts} · {info.get('chunk_count', '?')} chunks")


# ── Video loader ──────────────────────────────────────────────────────────────
with st.form("video_form"):
    url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    load_btn = st.form_submit_button("Load Video", type="primary")

if load_btn and url:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Add it to your .env file or Streamlit secrets.")
    else:
        with st.spinner("Fetching transcript and building index…"):
            try:
                chunks, video_id = get_transcript_chunks(url)

                existing = st.session_state.videos.get(video_id, {})
                chat_history = existing.get("chat_history", [])

                vectorstore, bm25, docs = get_or_build_vectorstore(chunks, video_id)

                summary = existing.get("summary") or generate_summary(
                    docs, model=selected_model
                )

                word_count = sum(len(c["text"].split()) for c in chunks)
                st.session_state.videos[video_id] = {
                    "vectorstore": vectorstore,
                    "bm25": bm25,
                    "docs": docs,
                    "url": url,
                    "summary": summary,
                    "chat_history": chat_history,
                    "word_count": word_count,
                }
                st.session_state.active_video_id = video_id
                st.success(f"✅ Loaded! {word_count:,} words · {len(docs)} chunks")

            except NoCaptionsError as e:
                st.warning(f"⚠️ {e}")
                st.info(
                    "**This video has no YouTube captions.** "
                    "Try a video that has captions enabled. Most popular English "
                    "YouTube videos have auto-generated captions.\n\n"
                    "**These always work:**\n"
                    "- TED Talks: youtube.com/tedtalks\n"
                    "- Kurzgesagt, Veritasium, 3Blue1Brown\n"
                    "- Khan Academy, MIT OpenCourseWare\n"
                    "- Any major news channel (BBC, CNN, etc.)"
                )

            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Failed to load video")


# ── Chat area ─────────────────────────────────────────────────────────────────
av = active_video()

if av:
    with st.expander("📋 Video Summary", expanded=True):
        st.markdown(av["summary"])

    st.divider()

    for turn in av["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])

    user_query = st.chat_input("Ask anything about the video…")

    if user_query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set.")
        else:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    source_docs = retrieve_and_rerank(
                        av["vectorstore"], av["bm25"], av["docs"], user_query
                    )
                    context = format_docs_with_timestamps(source_docs)
                    response = st.write_stream(
                        stream_response(
                            context, user_query, av["chat_history"],
                            model=selected_model,
                        )
                    )
                    render_sources(source_docs)

                    av["chat_history"].append({"role": "user", "content": user_query})
                    av["chat_history"].append({
                        "role": "assistant",
                        "content": response,
                        "sources": source_docs,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Generation failed")

    if av["chat_history"]:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                av["chat_history"] = []
                st.rerun()
        with c2:
            md = export_chat_markdown(av["url"], av["chat_history"])
            st.download_button(
                "⬇️ Export Chat", data=md,
                file_name="chat_export.md", mime="text/markdown",
                use_container_width=True,
            )

else:
    st.info("⬆️ Paste a YouTube URL above and click **Load Video** to start.")
