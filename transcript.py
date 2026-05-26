import logging
import re

from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class NoCaptionsError(RuntimeError):
    """Raised when a video has no YouTube captions available."""
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds → MM:SS or H:MM:SS."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _snippets_to_chunks(snippets: list, video_id: str) -> list[dict]:
    """Group raw transcript snippets into character-limited, timestamp-aware chunks."""
    if not snippets:
        raise NoCaptionsError("Transcript is empty.")

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = snippets[0]["start"]
    current_len: int = 0

    for snippet in snippets:
        text = snippet["text"].strip().replace("\n", " ")
        if not text:
            continue

        if current_len + len(text) > CHUNK_SIZE and current_texts:
            chunks.append({
                "text": " ".join(current_texts),
                "start": current_start,
                "end": snippet["start"],
            })
            overlap = " ".join(current_texts)[-CHUNK_OVERLAP:]
            current_texts = [overlap, text] if overlap else [text]
            current_start = snippet["start"]
            current_len = sum(len(t) for t in current_texts)
        else:
            current_texts.append(text)
            current_len += len(text)

    if current_texts:
        last = snippets[-1]
        chunks.append({
            "text": " ".join(current_texts),
            "start": current_start,
            "end": last["start"] + last.get("duration", 0),
        })

    logger.info(f"Built {len(chunks)} chunks for video {video_id}")
    return chunks


# ── YouTube caption fetch ─────────────────────────────────────────────────────

def _fetch_snippets(video_id: str) -> list:
    """
    Try every available strategy to get captions from YouTube.
      1. Manual English transcript
      2. Auto-generated English transcript
      3. Any available transcript in any language
      4. Plain api.fetch() as last resort
    Raises NoCaptionsError if nothing is found.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        all_transcripts = list(transcript_list)

        # 1. Manual English
        for t in all_transcripts:
            if not t.is_generated and t.language_code.startswith("en"):
                snippets = list(t.fetch())
                if snippets:
                    logger.info("Manual English transcript found")
                    return snippets

        # 2. Auto-generated English
        for t in all_transcripts:
            if t.is_generated and t.language_code.startswith("en"):
                snippets = list(t.fetch())
                if snippets:
                    logger.info("Auto-generated English transcript found")
                    return snippets

        # 3. Any transcript in any language
        for t in all_transcripts:
            try:
                snippets = list(t.fetch())
                if snippets:
                    logger.info(f"Using transcript: lang={t.language_code}")
                    return snippets
            except Exception:
                continue

    except Exception as e:
        logger.debug(f"list_transcripts failed: {e}")

    # 4. Plain fetch fallback
    try:
        api = YouTubeTranscriptApi()
        snippets = list(api.fetch(video_id))
        if snippets:
            logger.info("Fetched via plain api.fetch()")
            return snippets
    except Exception as e:
        logger.debug(f"Plain fetch failed: {e}")

    raise NoCaptionsError(
        "This video has no YouTube captions available. "
        "Please try a video that has captions enabled."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_transcript_chunks(url: str) -> tuple[list[dict], str]:
    """
    Fetch YouTube captions and split into timestamp-aware chunks.

    Returns:
        chunks   – list of {"text", "start", "end"}
        video_id – YouTube video ID

    Raises NoCaptionsError if the video has no captions.
    """
    video_id = extract_video_id(url)
    snippets = _fetch_snippets(video_id)
    chunks = _snippets_to_chunks(snippets, video_id)
    return chunks, video_id
