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


# ── VTT parser (for yt-dlp subtitle files) ───────────────────────────────────

def _vtt_time_to_seconds(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = "0", parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def _parse_vtt(content: str) -> list[dict]:
    """
    Parse a WebVTT subtitle file into snippet dicts with text/start/duration.
    Handles YouTube's auto-generated VTT format which has repeated lines and
    HTML timing tags like <00:00:01.000><c>word</c>.
    """
    snippets: list[dict] = []
    seen_texts: set[str] = set()  # de-duplicate consecutive identical lines

    # Split on blank lines to get cue blocks
    blocks = re.split(r"\n{2,}", content)
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Find the timestamp line
        ts_line = None
        ts_idx = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                ts_idx = i
                break
        if ts_line is None:
            continue

        # Parse start/end times
        ts_match = re.match(
            r"([\d:]+\.[\d]+)\s*-->\s*([\d:]+\.[\d]+)", ts_line
        )
        if not ts_match:
            continue
        start_sec = _vtt_time_to_seconds(ts_match.group(1))
        end_sec = _vtt_time_to_seconds(ts_match.group(2))

        # Everything after the timestamp line is the caption text
        raw_text = " ".join(lines[ts_idx + 1:])
        # Strip all VTT/HTML tags (<c>, <00:00:01.000>, </c>, etc.)
        text = re.sub(r"<[^>]+>", "", raw_text).strip()
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text)

        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        snippets.append({
            "text": text,
            "start": start_sec,
            "duration": end_sec - start_sec,
        })

    return snippets


# ── yt-dlp subtitle fetch ─────────────────────────────────────────────────────

def _fetch_via_ytdlp(video_id: str, cookie_path: str | None = None) -> list[dict]:
    """
    Download the subtitle/caption file for a video using yt-dlp (no audio).
    Prefers English; falls back to any available language.
    Returns snippet dicts with text/start/duration.
    Raises NoCaptionsError on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        raise NoCaptionsError("yt-dlp is not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "%(id)s")

        ydl_opts = {
            # No audio/video download
            "skip_download": True,
            # Fetch both manual and auto-generated subtitles
            "writesubtitles": True,
            "writeautomaticsub": True,
            # Request English first; yt-dlp falls back automatically
            "subtitleslangs": ["en", "en.*", "hi", "all"],
            "subtitlesformat": "vtt",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }
        if cookie_path:
            ydl_opts["cookiefile"] = cookie_path

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise NoCaptionsError(f"yt-dlp failed: {e}")

        # Pick the best downloaded .vtt file
        vtt_files = [f for f in os.listdir(tmpdir) if f.endswith(".vtt")]
        if not vtt_files:
            raise NoCaptionsError("yt-dlp: no subtitle files were downloaded")

        # Prefer English subtitles
        en_files = [f for f in vtt_files if re.search(r"\.(en|en-[A-Za-z]+)\.", f)]
        chosen = en_files[0] if en_files else vtt_files[0]
        logger.info(f"yt-dlp subtitle file: {chosen}")

        with open(os.path.join(tmpdir, chosen), "r", encoding="utf-8") as fh:
            vtt_content = fh.read()

        snippets = _parse_vtt(vtt_content)
        if not snippets:
            raise NoCaptionsError("yt-dlp: subtitle file was empty or unparseable")

        return snippets


# ── YouTube transcript API fetch ──────────────────────────────────────────────

def _fetch_snippets(video_id: str) -> list:
    """
    Try every available strategy to get captions from YouTube.

    Strategies 1-5 use youtube-transcript-api.
    Strategy 6 uses yt-dlp (subtitle-only download, no audio, no ffmpeg needed).
    Each strategy is isolated so one failure never silences the next.
    Raises NoCaptionsError with full diagnostics if nothing works.
    """
    errors: list[str] = []
    all_transcripts = []
    cookies = _cookie_path

    # ── List available transcripts ────────────────────────────────────────────
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

    # ── Strategy 1: Manual English ────────────────────────────────────────────
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

    # ── Strategy 2: Auto-generated English ───────────────────────────────────
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

    # ── Strategy 3: Any language (direct fetch) ───────────────────────────────
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

    # ── Strategy 4: get_transcript() with each known language code ────────────
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

    # ── Strategy 5: Translate any translatable transcript to English ──────────
    for t in all_transcripts:
        if not getattr(t, "is_translatable", False):
            continue
        try:
            snippets = list(t.translate("en").fetch())
            if snippets:
                logger.info(f"Strategy 5 success: translated {t.language_code} → en")
                return snippets
        except Exception as e:
            msg = f"{t.language_code}→en translate → {type(e).__name__}: {e}"
            logger.warning(msg)
            errors.append(msg)

    # ── Strategy 6: yt-dlp subtitle download (most robust) ───────────────────
    # youtube-transcript-api returns empty bodies when cookies are used from a
    # different IP (YouTube anti-bot measure). yt-dlp handles auth correctly.
    try:
        snippets = _fetch_via_ytdlp(video_id, cookie_path=cookies)
        if snippets:
            logger.info(f"Strategy 6 success: yt-dlp ({len(snippets)} snippets)")
            return snippets
    except NoCaptionsError as e:
        msg = f"yt-dlp → {e}"
        logger.warning(msg)
        errors.append(msg)
    except Exception as e:
        msg = f"yt-dlp → {type(e).__name__}: {e}"
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
