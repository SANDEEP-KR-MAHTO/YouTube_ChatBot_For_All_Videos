import glob
import logging
import os
import re
import shutil
import tempfile
from types import SimpleNamespace

from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_OVERLAP, CHUNK_SIZE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)


# ── YouTube cookies (needed on cloud deployments) ─────────────────────────────

def _get_cookies_file() -> str | None:
    """
    Write YouTube cookies from the environment to a temp file for yt-dlp.

    On Streamlit Cloud, set the secret YOUTUBE_COOKIES to the full contents
    of a Netscape-format cookies.txt exported from your browser while logged
    in to YouTube.  Locally, add the same variable to your .env file.

    Returns the path to the temp file, or None if no cookies are configured.
    The caller is responsible for deleting the file after use.
    """
    cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not cookies:
        return None
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="yt_cookies_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(cookies)
    logger.info("Using YouTube cookies from environment for yt-dlp")
    return path


# ── ffmpeg auto-detection ─────────────────────────────────────────────────────

def _patch_whisper_ffmpeg(ffmpeg_exe: str) -> None:
    """
    Replace whisper.audio.load_audio with a version that uses the absolute
    ffmpeg path instead of relying on PATH lookup.

    This is necessary on Windows where os.environ["PATH"] changes do not
    reliably propagate to subprocesses already in flight.
    """
    try:
        import numpy as np
        import whisper.audio as _wa
        from subprocess import CalledProcessError, run

        _sr = _wa.SAMPLE_RATE

        def _load_audio(file: str, sr: int = _sr) -> "np.ndarray":
            cmd = [
                ffmpeg_exe, "-nostdin", "-threads", "0",
                "-i", file,
                "-f", "s16le", "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-",
            ]
            try:
                out = run(cmd, capture_output=True, check=True).stdout
            except CalledProcessError as e:
                raise RuntimeError(
                    f"ffmpeg failed to decode audio: {e.stderr.decode()}"
                ) from e
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        _wa.load_audio = _load_audio
        logger.info(f"Patched whisper.audio.load_audio → {ffmpeg_exe}")
    except Exception as e:
        logger.warning(f"Could not patch whisper ffmpeg path: {e}")


def _ensure_ffmpeg() -> str | None:
    """
    Locate ffmpeg and make it available to both yt-dlp and Whisper.

    Priority:
      1. System ffmpeg already on PATH  → return None
      2. imageio-ffmpeg bundled binary  → patch Whisper to use it, return path
      3. Neither available              → raise RuntimeError with install hint
    """
    import shutil as _shutil
    if _shutil.which("ffmpeg"):
        return None  # system ffmpeg is fine; both yt-dlp and Whisper will find it

    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        # PATH update (helps yt-dlp on some systems)
        ffmpeg_dir = os.path.dirname(exe)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        # Patch Whisper directly — reliable on Windows where PATH updates
        # don't always propagate to subprocesses.
        _patch_whisper_ffmpeg(exe)
        logger.info(f"Using bundled ffmpeg from imageio-ffmpeg: {exe}")
        return exe
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found and imageio-ffmpeg is not installed.\n"
            "Fix (no manual install needed):  pip install imageio-ffmpeg\n"
            "Or manually: https://ffmpeg.org → add ffmpeg/bin to PATH."
        )


# ── Whisper model singleton ───────────────────────────────────────────────────
_whisper_cache: dict = {}


def _get_whisper_model(model_size: str):
    if model_size not in _whisper_cache:
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "openai-whisper is not installed. Run:\n"
                "  pip install openai-whisper\n"
                "Also make sure ffmpeg is on your PATH."
            )
        logger.info(f"Loading Whisper model '{model_size}' (downloads once on first use)")
        _whisper_cache[model_size] = whisper.load_model(model_size)
    return _whisper_cache[model_size]


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
        raise RuntimeError("Transcript is empty.")

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = snippets[0].start
    current_len: int = 0

    for snippet in snippets:
        text = snippet.text.strip().replace("\n", " ")
        if not text:
            continue

        if current_len + len(text) > CHUNK_SIZE and current_texts:
            chunks.append(
                {
                    "text": " ".join(current_texts),
                    "start": current_start,
                    "end": snippet.start,
                }
            )
            overlap = " ".join(current_texts)[-CHUNK_OVERLAP:]
            current_texts = [overlap, text] if overlap else [text]
            current_start = snippet.start
            current_len = sum(len(t) for t in current_texts)
        else:
            current_texts.append(text)
            current_len += len(text)

    if current_texts:
        last = snippets[-1]
        chunks.append(
            {
                "text": " ".join(current_texts),
                "start": current_start,
                "end": last.start + getattr(last, "duration", 0),
            }
        )

    logger.info(f"Built {len(chunks)} chunks for video {video_id}")
    return chunks


# ── YouTube transcript fetch ───────────────────────────────────────────────────

def _fetch_youtube_snippets(video_id: str, preferred_langs: list[str]) -> list | None:
    """
    Try to fetch YouTube captions with language preference.
    Returns a list of snippet objects, or None if unavailable.

    Strategy:
      1. list_transcripts → find manual transcript in preferred langs
      2. list_transcripts → find auto-generated transcript in preferred langs
      3. list_transcripts → take whatever is available (first transcript)
      4. Plain api.fetch() as final fallback
    """
    # ── Attempt 1-3: language-aware via list_transcripts ──────────────────────
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None
        # Try manual transcript in preferred languages
        try:
            transcript = transcript_list.find_manually_created_transcript(preferred_langs)
            logger.info(f"Found manual transcript: lang={transcript.language_code}")
        except Exception:
            pass

        # Try auto-generated in preferred languages
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(preferred_langs)
                logger.info(f"Found auto-generated transcript: lang={transcript.language_code}")
            except Exception:
                pass

        # Take the first available transcript regardless of language
        if transcript is None:
            try:
                transcript = next(iter(transcript_list))
                logger.info(
                    f"No preferred-language transcript found; "
                    f"using first available: lang={transcript.language_code}"
                )
            except StopIteration:
                pass

        if transcript is not None:
            return list(transcript.fetch())

    except Exception as e:
        logger.debug(f"list_transcripts approach failed: {e}")

    # ── Attempt 4: plain fetch (gets the default transcript) ──────────────────
    try:
        api = YouTubeTranscriptApi()
        snippets = list(api.fetch(video_id))
        logger.info("Fetched transcript via plain api.fetch()")
        return snippets
    except Exception as e:
        logger.debug(f"Plain fetch failed: {e}")

    return None


# ── Whisper fallback ──────────────────────────────────────────────────────────

def _whisper_transcribe(video_id: str, language: str | None, model_size: str) -> list:
    """
    Download YouTube audio with yt-dlp and transcribe locally with Whisper.
    Returns snippet-like objects compatible with _snippets_to_chunks().

    Requirements: pip install openai-whisper yt-dlp imageio-ffmpeg
                  (imageio-ffmpeg auto-downloads ffmpeg; no manual install needed)
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. Run:\n  pip install yt-dlp"
        )

    # Ensure ffmpeg is available (needed by Whisper for audio decoding).
    # imageio-ffmpeg provides only ffmpeg — NOT ffprobe.
    # We therefore skip yt-dlp's FFmpegExtractAudio postprocessor (which needs
    # both) and download the audio in its native container format instead.
    # Whisper can decode m4a/webm/opus directly using just ffmpeg.
    ffmpeg_exe = _ensure_ffmpeg()

    tmpdir = tempfile.mkdtemp()
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        out_template = os.path.join(tmpdir, f"{video_id}.%(ext)s")

        # Base yt-dlp options
        base_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            # No FFmpegExtractAudio postprocessor — imageio-ffmpeg has no ffprobe.
            # We hand the native m4a/webm file straight to Whisper.
            "quiet": True,
            "no_warnings": True,
        }
        if ffmpeg_exe:
            base_opts["ffmpeg_location"] = os.path.dirname(ffmpeg_exe)

        cookies_file = _get_cookies_file()
        if cookies_file:
            base_opts["cookiefile"] = cookies_file

        # YouTube periodically blocks the default (web) player client.
        # Try multiple player clients in order — ios and android use different
        # API endpoints that are typically not blocked.
        player_clients = ["ios", "android", "web"]

        import yt_dlp

        last_error: Exception | None = None
        downloaded = False

        for client in player_clients:
            ydl_opts = {
                **base_opts,
                "extractor_args": {"youtube": {"player_client": [client]}},
            }
            logger.info(f"Attempting yt-dlp download for {video_id} (player_client={client})")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                downloaded = True
                break
            except Exception as e:
                last_error = e
                logger.warning(f"yt-dlp player_client={client} failed: {e}")
                # Clean up any partial files before retrying
                for f in glob.glob(os.path.join(tmpdir, f"{video_id}.*")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        if not downloaded:
            err = str(last_error)
            if any(k in err for k in ("403", "Forbidden", "Sign in", "bot", "cookies")):
                raise RuntimeError(
                    "YouTube blocked the audio download (HTTP 403 Forbidden) "
                    "on all player clients (ios, android, web).\n\n"
                    "Fix: export your YouTube cookies and set them as "
                    "YOUTUBE_COOKIES in your .env file (locally) or Streamlit "
                    "secrets (on cloud).\n"
                    "See README → 'Deploying to Streamlit Cloud' for instructions."
                ) from None
            raise last_error

        # Locate the downloaded file (extension varies: m4a, webm, opus, …)
        audio_files = glob.glob(os.path.join(tmpdir, f"{video_id}.*"))
        if not audio_files:
            raise RuntimeError("yt-dlp downloaded nothing — check your internet connection.")
        audio_path = audio_files[0]

        # Transcribe
        model = _get_whisper_model(model_size)
        whisper_kwargs: dict = {"task": "transcribe", "verbose": False}
        if language:
            whisper_kwargs["language"] = language

        logger.info(
            f"Transcribing with Whisper ({model_size}), "
            f"language={'auto' if language is None else language}"
        )
        result = model.transcribe(audio_path, **whisper_kwargs)

        detected = result.get("language", language or "?")
        segments = result.get("segments", [])
        logger.info(
            f"Whisper finished: {len(segments)} segments, "
            f"detected_language={detected}"
        )

        # Wrap segments as snippet-like objects
        snippets = [
            SimpleNamespace(
                text=seg["text"].strip(),
                start=seg["start"],
                duration=seg["end"] - seg["start"],
            )
            for seg in segments
            if seg["text"].strip()
        ]
        return snippets

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if cookies_file and os.path.exists(cookies_file):
            os.unlink(cookies_file)


# ── Public API ────────────────────────────────────────────────────────────────

def get_transcript_chunks(
    url: str,
    language: str | None = None,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
) -> tuple[list[dict], str, str]:
    """
    Fetch transcript and split into timestamp-aware chunks.

    Args:
        url               – YouTube video URL
        language          – ISO-639-1 code (e.g. "hi", "en") or None for auto
        whisper_model_size – Whisper model size used if YouTube captions are absent

    Returns:
        chunks       – list of {"text", "start", "end"}
        video_id     – YouTube video ID
        source       – "youtube" | "whisper"
    """
    video_id = extract_video_id(url)
    preferred_langs = ([language] if language else []) + ["hi", "en"]

    # ── Try YouTube captions first ────────────────────────────────────────────
    snippets = _fetch_youtube_snippets(video_id, preferred_langs)

    if snippets:
        chunks = _snippets_to_chunks(snippets, video_id)
        return chunks, video_id, "youtube"

    # ── Fall back to Whisper ──────────────────────────────────────────────────
    logger.info(
        f"No YouTube captions for {video_id}; falling back to Whisper ({whisper_model_size})"
    )
    snippets = _whisper_transcribe(video_id, language, whisper_model_size)
    chunks = _snippets_to_chunks(snippets, video_id)
    return chunks, video_id, "whisper"


def get_transcript(url: str) -> tuple[str, str]:
    """Backward-compatible: returns (full_text, video_id)."""
    chunks, video_id, _ = get_transcript_chunks(url)
    return " ".join(c["text"] for c in chunks), video_id
