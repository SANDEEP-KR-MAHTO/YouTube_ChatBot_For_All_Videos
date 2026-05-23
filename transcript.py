import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

from youtube_transcript_api import YouTubeTranscriptApi

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

# Groq's hard upload limit
_GROQ_MAX_MB = 24.5
# Split audio into chunks of this many minutes (well under the size limit)
_SPLIT_MINUTES = 20


# ── YouTube cookies (needed when YouTube blocks cloud IPs) ────────────────────

def _get_cookies_file() -> str | None:
    cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not cookies:
        return None
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="yt_cookies_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(cookies)
    logger.info("Using YouTube cookies from environment for yt-dlp")
    return path


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


# ── YouTube caption fetch ─────────────────────────────────────────────────────

def _fetch_youtube_snippets(video_id: str, preferred_langs: list[str]) -> list | None:
    """
    Try to fetch YouTube captions with language preference.
    Returns snippet objects or None if unavailable.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(preferred_langs)
            logger.info(f"Found manual transcript: lang={transcript.language_code}")
        except Exception:
            pass

        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(preferred_langs)
                logger.info(f"Found auto-generated transcript: lang={transcript.language_code}")
            except Exception:
                pass

        if transcript is None:
            try:
                transcript = next(iter(transcript_list))
                logger.info(f"Using first available transcript: lang={transcript.language_code}")
            except StopIteration:
                pass

        if transcript is not None:
            return list(transcript.fetch())

    except Exception as e:
        logger.debug(f"list_transcripts failed: {e}")

    try:
        api = YouTubeTranscriptApi()
        snippets = list(api.fetch(video_id))
        logger.info("Fetched transcript via plain api.fetch()")
        return snippets
    except Exception as e:
        logger.debug(f"Plain fetch failed: {e}")

    return None


# ── Audio download via yt-dlp ─────────────────────────────────────────────────

def _download_audio(video_id: str, out_dir: str) -> str:
    """
    Download the lowest-bitrate audio available for a YouTube video.

    Using worstaudio (typically 48–64 kbps) keeps most videos under Groq's
    25 MB limit:
        48 kbps × 1 h  =  ~21 MB  ✓
        48 kbps × 90 m =  ~32 MB  → auto-split before sending to Groq
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = os.path.join(out_dir, f"{video_id}.%(ext)s")

    # Prefer the smallest audio format available.
    # worstaudio picks the lowest-bitrate audio-only stream — usually webm/opus
    # at ~48-64 kbps, which is plenty for speech transcription.
    # Explicit ext preferences ensure we get a container Groq can decode
    # without ffmpeg post-processing.
    format_string = (
        "worstaudio[ext=webm]"
        "/worstaudio[ext=m4a]"
        "/worstaudio"
        "/bestaudio[ext=m4a]"
        "/bestaudio[ext=webm]"
        "/bestaudio"
        "/best[height<=144]"
        "/best"
    )

    base_opts: dict = {
        "format": format_string,
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
    }

    cookies_file = _get_cookies_file()
    if cookies_file:
        base_opts["cookiefile"] = cookies_file

    # mweb (mobile web) often bypasses datacenter-IP blocks that affect the
    # desktop web client. Try it first, then fall back in order.
    player_clients = ["mweb", "ios", "android", "web"]

    last_error: Exception | None = None

    try:
        for client in player_clients:
            opts = {
                **base_opts,
                "extractor_args": {
                    "youtube": {
                        "player_client": [client],
                        "skip": ["hls"],  # HLS streams require ffmpeg to merge; skip
                    }
                },
            }
            logger.info(f"yt-dlp download attempt: player_client={client}")
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])

                audio_files = glob.glob(os.path.join(out_dir, f"{video_id}.*"))
                if audio_files:
                    logger.info(f"Downloaded: {audio_files[0]}")
                    return audio_files[0]

            except Exception as exc:
                last_error = exc
                logger.warning(f"yt-dlp player_client={client} failed: {exc}")
                for f in glob.glob(os.path.join(out_dir, f"{video_id}.*")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    finally:
        if cookies_file and os.path.exists(cookies_file):
            try:
                os.unlink(cookies_file)
            except OSError:
                pass

    err = str(last_error or "")
    if any(k in err for k in ("403", "Forbidden", "Sign in", "bot", "cookies", "blocked")):
        raise RuntimeError(
            "YouTube blocked the audio download from this server's IP address.\n\n"
            "This video has no YouTube captions, so the app tried to download its "
            "audio for Groq Whisper transcription — but YouTube blocks cloud datacenter IPs.\n\n"
            "Fix: export your YouTube cookies and add them as YOUTUBE_COOKIES in "
            "Streamlit secrets (see README for step-by-step instructions)."
        )
    if last_error:
        raise RuntimeError(f"Audio download failed: {last_error}") from last_error
    raise RuntimeError("Audio download failed: no file was produced.")


# ── ffmpeg resolution ─────────────────────────────────────────────────────────

def _get_ffmpeg_exe() -> str:
    """
    Return a path to an ffmpeg executable.

    Priority:
      1. System ffmpeg already on PATH  (works on Streamlit Cloud via packages.txt)
      2. imageio-ffmpeg bundled binary  (works locally — no manual install needed)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"Using bundled ffmpeg from imageio-ffmpeg: {exe}")
        return exe
    except ImportError:
        pass

    raise RuntimeError(
        "ffmpeg not found.\n"
        "It is installed automatically on Streamlit Cloud via packages.txt.\n"
        "Locally, run:  pip install imageio-ffmpeg  (no manual install needed)."
    )


# ── Audio re-encoding + splitting (for files > Groq's 25 MB limit) ───────────

def _reencode_audio(audio_path: str, out_dir: str) -> str:
    """
    Re-encode audio to 32 kbps mono mp3.

    At 32 kbps a 20-minute chunk is ~5 MB — well inside Groq's 25 MB limit
    regardless of the original download quality.
    """
    ffmpeg = _get_ffmpeg_exe()
    out_path = os.path.join(out_dir, "reencoded.mp3")
    subprocess.run(
        [
            ffmpeg, "-i", audio_path,
            "-ac", "1",        # mono
            "-ab", "32k",      # 32 kbps — adequate for speech transcription
            "-y", out_path,
        ],
        check=True,
        capture_output=True,
    )
    new_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Re-encoded audio to 32 kbps mono: {new_mb:.1f} MB")
    return out_path


def _split_audio(audio_path: str, out_dir: str, chunk_minutes: int = _SPLIT_MINUTES) -> list[str]:
    """
    Split an audio file into fixed-duration chunks using ffmpeg.
    Returns sorted list of chunk file paths.
    """
    ffmpeg = _get_ffmpeg_exe()
    ext = os.path.splitext(audio_path)[1]
    chunk_pattern = os.path.join(out_dir, f"chunk_%03d{ext}")

    subprocess.run(
        [
            ffmpeg, "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(chunk_minutes * 60),
            "-c", "copy",
            "-reset_timestamps", "1",
            chunk_pattern,
        ],
        check=True,
        capture_output=True,
    )

    chunks = sorted(glob.glob(os.path.join(out_dir, f"chunk_*{ext}")))
    if not chunks:
        raise RuntimeError("ffmpeg produced no audio chunks.")
    logger.info(f"Split audio into {len(chunks)} chunks of ≤{chunk_minutes} min")
    return chunks


# ── Groq Whisper transcription (free, cloud-safe) ────────────────────────────

def _groq_call(audio_path: str, language: str | None) -> list:
    """Send a single audio file to Groq Whisper. File must be < 25 MB."""
    from groq import Groq

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > _GROQ_MAX_MB:
        raise RuntimeError(
            f"Audio chunk is {file_size_mb:.1f} MB which still exceeds Groq's "
            f"{_GROQ_MAX_MB} MB limit after re-encoding. "
            "This video may have an unusually high audio bitrate. "
            "Try reducing _SPLIT_MINUTES in transcript.py (e.g. to 10)."
        )

    client = Groq()
    logger.info(f"Groq Whisper: {os.path.basename(audio_path)} ({file_size_mb:.1f} MB)")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    kwargs: dict = {
        "file": (os.path.basename(audio_path), audio_bytes),
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language:
        kwargs["language"] = language

    transcription = client.audio.transcriptions.create(**kwargs)
    raw_segments = getattr(transcription, "segments", None) or []

    # Groq SDK may return segments as dicts or as attribute-bearing objects
    # depending on the SDK version. Normalise to SimpleNamespace so callers
    # can always use seg.text / seg.start / seg.end uniformly.
    segments = []
    for seg in raw_segments:
        if isinstance(seg, dict):
            segments.append(SimpleNamespace(**seg))
        else:
            segments.append(seg)

    return segments


def _groq_transcribe(audio_path: str, language: str | None) -> list:
    """
    Transcribe audio using Groq's free Whisper API.

    If the file exceeds Groq's 25 MB limit the audio is split into
    20-minute chunks with ffmpeg, each chunk is transcribed separately,
    and timestamps are stitched back into a continuous sequence.

    This makes any video length work — no manual splitting required.
    """
    try:
        from groq import Groq  # noqa: F401
    except ImportError:
        raise RuntimeError("groq package is not installed. Run: pip install groq")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    # ── Small file: send directly ─────────────────────────────────────────────
    if file_size_mb <= _GROQ_MAX_MB:
        segments = _groq_call(audio_path, language)
        return [
            SimpleNamespace(
                text=seg.text.strip(),
                start=seg.start,
                duration=seg.end - seg.start,
            )
            for seg in segments
            if seg.text.strip()
        ]

    # ── Large file: re-encode to 32 kbps → split → transcribe → stitch ─────────
    logger.info(
        f"Audio is {file_size_mb:.1f} MB > {_GROQ_MAX_MB} MB — "
        f"re-encoding to 32 kbps then splitting into {_SPLIT_MINUTES}-min chunks"
    )

    _get_ffmpeg_exe()  # raises early with a clear message if ffmpeg is missing

    split_dir = tempfile.mkdtemp()
    try:
        # Re-encode first so every chunk is guaranteed to be tiny (~5 MB per 20 min)
        reencoded_path = _reencode_audio(audio_path, split_dir)
        chunk_paths = _split_audio(reencoded_path, split_dir)

        all_snippets: list = []
        time_offset: float = 0.0

        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            segments = _groq_call(chunk_path, language)

            chunk_end = 0.0
            for seg in segments:
                if not seg.text.strip():
                    continue
                all_snippets.append(
                    SimpleNamespace(
                        text=seg.text.strip(),
                        start=seg.start + time_offset,
                        duration=seg.end - seg.start,
                    )
                )
                chunk_end = max(chunk_end, seg.end)

            # Advance offset by the actual spoken duration of this chunk
            time_offset += chunk_end if chunk_end > 0 else _SPLIT_MINUTES * 60

        return all_snippets

    finally:
        shutil.rmtree(split_dir, ignore_errors=True)


# ── Public API ────────────────────────────────────────────────────────────────

def get_transcript_chunks(
    url: str,
    language: str | None = None,
) -> tuple[list[dict], str, str]:
    """
    Fetch transcript and split into timestamp-aware chunks.

    Strategy:
      1. YouTube Transcript API (captions) — instant, no quota used
      2. yt-dlp audio download → Groq Whisper API (free) — for caption-free videos
         Large audio is auto-split into 20-min chunks before sending to Groq.

    Args:
        url      – YouTube video URL
        language – ISO-639-1 code (e.g. "hi", "en") or None for auto-detect

    Returns:
        chunks   – list of {"text", "start", "end"}
        video_id – YouTube video ID
        source   – "youtube" | "groq_whisper"
    """
    video_id = extract_video_id(url)
    preferred_langs = ([language] if language else []) + ["hi", "en"]

    # ── 1. Try YouTube captions ───────────────────────────────────────────────
    snippets = _fetch_youtube_snippets(video_id, preferred_langs)
    if snippets:
        chunks = _snippets_to_chunks(snippets, video_id)
        return chunks, video_id, "youtube"

    # ── 2. Fall back to Groq Whisper ──────────────────────────────────────────
    logger.info(f"No YouTube captions for {video_id}; falling back to Groq Whisper")

    tmpdir = tempfile.mkdtemp()
    try:
        audio_path = _download_audio(video_id, tmpdir)
        snippets = _groq_transcribe(audio_path, language)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    chunks = _snippets_to_chunks(snippets, video_id)
    return chunks, video_id, "groq_whisper"


def get_transcript(url: str) -> tuple[str, str]:
    """Backward-compatible: returns (full_text, video_id)."""
    chunks, video_id, _ = get_transcript_chunks(url)
    return " ".join(c["text"] for c in chunks), video_id
