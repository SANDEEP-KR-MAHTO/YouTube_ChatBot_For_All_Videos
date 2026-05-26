import logging
import os
import re
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

# Path to a Netscape-format cookies.txt file written at startup by app.py.
# When set, all YouTube API calls include the cookie header so YouTube
# treats the request as coming from a logged-in browser instead of a bot.
_cookie_path: str | None = None


def init_cookies(cookie_content: str) -> None:
    """
    Write the raw cookies.txt content to a temp file and remember its path.
    Call this once at app startup if YOUTUBE_COOKIES is available in secrets.
    """
    global _cookie_path
    if not cookie_content or not cookie_content.strip():
        return
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(cookie_content)
    tmp.close()
    _cookie_path = tmp.name
    logger.info(f"YouTube cookies initialised ({len(cookie_content):,} bytes)")


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

    # Support both dict-style and object-style snippets across API versions
    def _get(snippet, key, default=0):
        if isinstance(snippet, dict):
            return snippet.get(key, default)
        return getattr(snippet, key, default)

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = _get(snippets[0], "start")
    current_len: int = 0

    for snippet in snippets:
        text = str(_get(snippet, "text", "")).strip().replace("\n", " ")
        if not text:
            continue

        snippet_start = _get(snippet, "start")
        if current_len + len(text) > CHUNK_SIZE and current_texts:
            chunks.append({
                "text": " ".join(current_texts),
                "start": current_start,
                "end": snippet_start,
            })
            overlap = " ".join(current_texts)[-CHUNK_OVERLAP:]
            current_texts = [overlap, text] if overlap else [text]
            current_start = snippet_start
            current_len = sum(len(t) for t in current_texts)
        else:
            current_texts.append(text)
            current_len += len(text)

    if current_texts:
        last = snippets[-1]
        chunks.append({
            "text": " ".join(current_texts),
            "start": current_start,
            "end": _get(last, "start") + _get(last, "duration"),
        })

    logger.info(f"Built {len(chunks)} chunks for video {video_id}")
    return chunks


# ── YouTube caption fetch ─────────────────────────────────────────────────────

def _fetch_snippets(video_id: str) -> list:
    """
    Try every available strategy to get captions from YouTube.
      1. Manual English transcript  (with cookies if available)
      2. Auto-generated English transcript
      3. Any available transcript in any language
      4. get_transcript() class-method fallback
    Each strategy is isolated so one failure never silences the next.
    Raises NoCaptionsError if nothing is found.
    """
    errors: list[str] = []
    all_transcripts = []
    cookies = _cookie_path  # may be None — API ignores it when None

    try:
        kw = {"cookies": cookies} if cookies else {}
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, **kw)
        all_transcripts = list(transcript_list)
        logger.info(
            f"list_transcripts: {len(all_transcripts)} transcripts for {video_id} — "
            + ", ".join(
                f"{t.language_code}({'auto' if t.is_generated else 'manual'})"
                for t in all_transcripts
            )
        )
    except Exception as e:
        msg = f"list_transcripts → {type(e).__name__}: {e}"
        logger.warning(msg)
        errors.append(msg)

    # 1. Manual English
    for t in all_transcripts:
        if not t.is_generated and t.language_code.startswith("en"):
            try:
                snippets = list(t.fetch())
                if snippets:
                    logger.info("Strategy 1 success: manual English transcript")
                    return snippets
            except Exception as e:
                msg = f"manual-en fetch → {type(e).__name__}: {e}"
                logger.warning(msg)
                errors.append(msg)

    # 2. Auto-generated English
    for t in all_transcripts:
        if t.is_generated and t.language_code.startswith("en"):
            try:
                snippets = list(t.fetch())
                if snippets:
                    logger.info("Strategy 2 success: auto-generated English transcript")
                    return snippets
            except Exception as e:
                msg = f"auto-en fetch → {type(e).__name__}: {e}"
                logger.warning(msg)
                errors.append(msg)

    # 3. Any transcript in any language (direct fetch)
    for t in all_transcripts:
        try:
            snippets = list(t.fetch())
            if snippets:
                logger.info(f"Strategy 3 success: lang={t.language_code}")
                return snippets
        except Exception as e:
            msg = f"{t.language_code} fetch → {type(e).__name__}: {e}"
            logger.warning(msg)
            errors.append(msg)

    # 4. get_transcript() using every language code we know about
    #    (strategy 3 may fail with ParseError on auto-generated tracks; this
    #    code path uses a different internal fetch mechanism and often succeeds)
    langs_to_try = ["en"] + [t.language_code for t in all_transcripts]
    seen: set[str] = set()
    for lang in langs_to_try:
        if lang in seen:
            continue
        seen.add(lang)
        try:
            kw = {"cookies": cookies} if cookies else {}
            snippets = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[lang], **kw
            )
            if snippets:
                logger.info(f"Strategy 4 success: get_transcript(languages=[{lang!r}])")
                return list(snippets)
        except Exception as e:
            msg = f"get_transcript({lang!r}) → {type(e).__name__}: {e}"
            logger.warning(msg)
            errors.append(msg)

    # 5. Translate any translatable transcript to English
    #    Useful when the video only has non-English auto-generated captions
    for t in all_transcripts:
        if not getattr(t, "is_translatable", False):
            continue
        try:
            snippets = list(t.translate("en").fetch())
            if snippets:
                logger.info(
                    f"Strategy 5 success: translated {t.language_code} → en"
                )
                return snippets
        except Exception as e:
            msg = f"{t.language_code}→en translate → {type(e).__name__}: {e}"
            logger.warning(msg)
            errors.append(msg)

    detail = " | ".join(errors) if errors else "no transcripts listed by YouTube"
    raise NoCaptionsError(
        f"Could not retrieve captions for this video.\n\nDiagnostic: {detail}"
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
