#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio==6.8.0",
#     "click==8.3.1",
#     "loguru==0.7.3",
#     "soundfile==0.13.1",
#     "numpy==2.4.2",
# ]
# ///
"""Collaborative multi-user audio dataset recorder for ASR fine-tuning.

A Gradio web UI where multiple volunteers record themselves speaking sentences
to build an ASR training dataset. Each user authenticates via a CSV-based
user list, gets assigned sentences from a shared pool, and recordings are
saved as 16 kHz mono WAV files with metadata in a NeMo-compatible JSONL.

Sentence assignment uses a file-lock-protected claims system so multiple
users can record simultaneously without conflicts.

Usage:
    uv run app.py sentences.jsonl --users-csv users.csv
    uv run app.py sentences.jsonl --users-csv users.csv --salt mysalt --port 7860
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import random
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import click
import gradio as gr
import soundfile as sf
from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
# Remove default loguru handler and configure both stderr + file logging.

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/app/logs/audiodataset.log", level="DEBUG", rotation="10 MB")

# ── Constants ─────────────────────────────────────────────────────────────────

# How many sentences to pre-assign to each user on login.
SENTENCES_PER_USER = 5

# Claims older than this (seconds) from a *different* user are considered stale.
# (Same-user stale claims are always cleaned up on login.)
CLAIM_TIMEOUT_S = 3600

# ── User identity ─────────────────────────────────────────────────────────────


def load_users_csv(csv_path: str) -> list[tuple[str, str]]:
    """Load ``username,password`` pairs from a CSV file (no header row).

    Parameters
    ----------
    csv_path : str
        Path to a CSV with two columns: username, password.

    Returns
    -------
    list[tuple[str, str]]
        List of (username, password) tuples for Gradio auth.
    """
    users: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                users.append((row[0].strip(), row[1].strip()))
    logger.info("Loaded {} users from {}", len(users), csv_path)
    return users


def make_userid(*, username: str, salt: str) -> str:
    """Deterministic 12-char hex userid from salted username hash.

    Parameters
    ----------
    username : str
        The plain-text username.
    salt : str
        A fixed salt string shared across sessions.

    Returns
    -------
    str
        First 12 hex characters of SHA-256(salt + username).
    """
    return hashlib.sha256((salt + username).encode()).hexdigest()[:12]


# ── File locking ──────────────────────────────────────────────────────────────


@contextmanager
def file_lock(lock_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock (fcntl.flock) for cross-process safety.

    Parameters
    ----------
    lock_path : Path
        Path to the lock file (created if it doesn't exist).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, "w")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


# ── Claims management ─────────────────────────────────────────────────────────
# Claims track which sentences are assigned to which user. This prevents two
# users from being assigned the same sentence simultaneously. Claims are stored
# in a JSON file protected by a file lock.
#
# Structure: { "<sentence_index>": {"username": "...", "timestamp": <epoch>} }


def _read_claims(claims_path: Path) -> dict[str, dict[str, Any]]:
    """Read claims file. Returns empty dict if file doesn't exist."""
    if not claims_path.exists():
        return {}
    with open(claims_path, encoding="utf-8") as f:
        return json.load(f)


def _write_claims(claims_path: Path, claims: dict[str, dict[str, Any]]) -> None:
    """Atomically write claims file."""
    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)


def _read_completed_indices(output_jsonl: Path) -> set[int]:
    """Read output JSONL and return set of sentence indices already recorded.

    Each completed line has a ``sentence_index`` field added at save time.
    """
    indices: set[int] = set()
    if not output_jsonl.exists():
        return indices
    with open(output_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "sentence_index" in entry:
                    indices.add(entry["sentence_index"])
            except json.JSONDecodeError:
                pass
    return indices


def assign_sentences(
    *,
    username: str,
    total_sentences: int,
    claims_path: Path,
    lock_path: Path,
    output_jsonl: Path,
    count: int = SENTENCES_PER_USER,
) -> list[int]:
    """Assign up to ``count`` unrecorded, unclaimed sentences to a user.

    Under file lock:
    1. Clean up stale claims from this user (page refresh scenario).
    2. Read completed indices from output JSONL.
    3. Find unclaimed, unrecorded indices.
    4. Assign random subset to user, write claims.

    Parameters
    ----------
    username : str
        The authenticated username.
    total_sentences : int
        Total number of sentences in the input pool.
    claims_path : Path
        Path to the claims JSON file.
    lock_path : Path
        Path to the lock file for synchronization.
    output_jsonl : Path
        Path to the output JSONL with completed recordings.
    count : int
        Number of sentences to assign.

    Returns
    -------
    list[int]
        Sentence indices assigned to this user (may be fewer than ``count``
        if the pool is nearly exhausted).
    """
    with file_lock(lock_path):
        claims = _read_claims(claims_path)
        completed = _read_completed_indices(output_jsonl)

        # Clean up stale claims from this user (e.g. page refresh).
        stale_keys = [
            k for k, v in claims.items()
            if v["username"] == username
        ]
        for k in stale_keys:
            del claims[k]
            logger.debug("Cleaned stale claim for user '{}': sentence {}", username, k)

        # Find available sentence indices: not completed and not claimed.
        claimed_indices = {int(k) for k in claims}
        available = [
            i for i in range(total_sentences)
            if i not in completed and i not in claimed_indices
        ]

        # Pick random subset.
        assigned = random.sample(available, min(count, len(available)))

        # Write new claims.
        now = time.time()
        for idx in assigned:
            claims[str(idx)] = {"username": username, "timestamp": now}
        _write_claims(claims_path, claims)

    logger.info(
        "Assigned {} sentences to '{}': {} (pool: {} available, {} completed, {} claimed)",
        len(assigned), username, assigned,
        len(available), len(completed), len(claimed_indices),
    )
    return assigned


def complete_and_reassign(
    *,
    username: str,
    sentence_index: int,
    total_sentences: int,
    claims_path: Path,
    lock_path: Path,
    output_jsonl: Path,
    completed_cache: set[int] | None = None,
) -> int | None:
    """Mark a sentence as completed and assign one new sentence.

    Under file lock:
    1. Remove the completed sentence from claims.
    2. Find one new unrecorded, unclaimed sentence.
    3. Claim it and return its index.

    Parameters
    ----------
    username : str
        The authenticated username.
    sentence_index : int
        Index of the just-completed sentence.
    total_sentences : int
        Total number of sentences in the input pool.
    claims_path : Path
        Path to the claims JSON file.
    lock_path : Path
        Path to the lock file.
    output_jsonl : Path
        Path to the output JSONL.

    Returns
    -------
    int | None
        Index of newly assigned sentence, or None if pool is exhausted.
    """
    with file_lock(lock_path):
        t_lock = time.time()
        claims = _read_claims(claims_path)
        t_claims = time.time()
        if completed_cache is not None:
            completed = completed_cache | {sentence_index}
        else:
            completed = _read_completed_indices(output_jsonl)
        t_completed = time.time()
        logger.debug(
            "complete_and_reassign lock timings: claims={:.3f}s, completed={:.3f}s (cache={})",
            t_claims - t_lock, t_completed - t_claims, completed_cache is not None,
        )

        # Remove the completed sentence's claim.
        claims.pop(str(sentence_index), None)

        # Find one new sentence.
        claimed_indices = {int(k) for k in claims}
        available = [
            i for i in range(total_sentences)
            if i not in completed and i not in claimed_indices
        ]

        new_idx = None
        if available:
            new_idx = random.choice(available)
            claims[str(new_idx)] = {"username": username, "timestamp": time.time()}

        _write_claims(claims_path, claims)

    if new_idx is not None:
        logger.debug("Reassigned sentence {} to '{}' (replacing {})", new_idx, username, sentence_index)
    else:
        logger.info("No more sentences to assign to '{}' (pool exhausted)", username)
    return new_idx



# ── Data loading ──────────────────────────────────────────────────────────────


def load_sentences(jsonl_path: str) -> list[dict[str, Any]]:
    """Load sentence entries from a JSONL file.

    Accepts both plain ``{"text": "..."}`` and NeMo format
    ``{"audio_filepath": "...", "text": "...", "duration": ...}``.
    The ``text`` field is required. If ``audio_filepath`` is present and
    the file exists (resolved relative to the JSONL directory), it is
    stored so the UI can display a reference audio player.

    Parameters
    ----------
    jsonl_path : str
        Path to the input JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts with keys ``"text"`` (str) and optionally
        ``"audio_filepath"`` (str, absolute path to a playable file).
    """
    jsonl_dir = Path(jsonl_path).resolve().parent
    # Build a recursive filename index for the fallback search.
    # NeMo manifests often use absolute paths that break when the dataset
    # is moved, so we build a {filename -> path} map by recursively listing
    # all files under the JSONL parent directory.
    logger.info("Building recursive file index under '{}' for audio fallback...", jsonl_dir.parent)
    _file_index: dict[str, Path] = {}
    try:
        for p in jsonl_dir.parent.rglob("*"):
            if p.is_file():
                # First match wins (closer to root is listed first by rglob).
                _file_index.setdefault(p.name, p)
    except PermissionError:
        logger.warning("Permission denied while scanning '{}'", jsonl_dir.parent)
    logger.info("File index built: {} files found", len(_file_index))

    def _resolve_audio(raw: str) -> Path | None:
        """Try to find an audio file, with progressive fallback.

        Resolution order:
        1. Path as-is (absolute or relative to CWD).
        2. Relative to JSONL directory.
        3. Filename-only search via recursive file index of the JSONL
           parent directory — handles NeMo absolute paths that became
           stale after moving the dataset.
        """
        p = Path(raw)
        fname = p.name
        # 1. Direct path (works for absolute or CWD-relative).
        if p.exists():
            logger.debug("Audio '{}': found at direct path", fname)
            return p.resolve()
        # 2. Relative to JSONL directory.
        rel = jsonl_dir / p
        if rel.exists():
            logger.debug("Audio '{}': found relative to JSONL dir", fname)
            return rel.resolve()
        # 3. Filename-only fallback via recursive index.
        match = _file_index.get(fname)
        if match is not None:
            logger.debug("Audio '{}': found via recursive search at '{}'", fname, match)
            return match.resolve()
        logger.warning("Audio '{}': NOT found (original path: '{}')", fname, raw)
        return None

    sentences: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if not text:
                    logger.warning("Line {} has no 'text' field, skipping", line_no)
                    continue
                entry: dict[str, Any] = {"text": text}
                # Resolve audio_filepath with fallback search so that
                # stale NeMo absolute paths still work after moving data.
                raw_audio = obj.get("audio_filepath", "")
                if raw_audio:
                    resolved = _resolve_audio(raw_audio)
                    if resolved is not None:
                        entry["audio_filepath"] = str(resolved)
                    else:
                        logger.debug("Line {}: audio '{}' not found in any fallback location", line_no, raw_audio)
                sentences.append(entry)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line {}: {}", line_no, exc)
    n_with_audio = sum(1 for s in sentences if "audio_filepath" in s)
    logger.info(
        "Loaded {} sentences from {} ({} with resolved audio)",
        len(sentences), jsonl_path, n_with_audio,
    )
    return sentences


# ── Audio processing ──────────────────────────────────────────────────────────


def _trim_silence(
    data: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -40.0,
    frame_ms: int = 10,
    min_silence_ms: int = 50,
) -> np.ndarray:
    """Trim leading and trailing silence from an audio signal.

    Uses short-time energy in small frames to detect where speech starts
    and ends. A frame is considered silent if its RMS energy is below
    ``threshold_db`` (relative to full-scale). Silence must last at least
    ``min_silence_ms`` to count, avoiding false trims on brief pauses.

    Parameters
    ----------
    data : np.ndarray
        Mono audio samples (float).
    sr : int
        Sample rate in Hz.
    threshold_db : float
        RMS threshold in dBFS below which a frame is silent.
    frame_ms : int
        Frame length in milliseconds for energy computation.
    min_silence_ms : int
        Minimum consecutive silence duration (ms) at the edges to trim.

    Returns
    -------
    np.ndarray
        Trimmed audio. Returns the original if no silence is found.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len == 0 or len(data) < frame_len:
        return data

    threshold_linear = 10 ** (threshold_db / 20)
    n_frames = len(data) // frame_len
    min_silent_frames = max(1, int(min_silence_ms / frame_ms))

    # Compute RMS energy per frame.
    frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Find first non-silent frame (from start).
    start_frame = 0
    for i in range(n_frames):
        if rms[i] >= threshold_linear:
            start_frame = i
            break
    else:
        # Entire signal is silent — return as-is to avoid empty output.
        return data

    # Find last non-silent frame (from end).
    end_frame = n_frames - 1
    for i in range(n_frames - 1, -1, -1):
        if rms[i] >= threshold_linear:
            end_frame = i
            break

    # Only trim if the silent region is long enough.
    if start_frame < min_silent_frames:
        start_frame = 0
    if (n_frames - 1 - end_frame) < min_silent_frames:
        end_frame = n_frames - 1

    start_sample = start_frame * frame_len
    end_sample = min((end_frame + 1) * frame_len, len(data))
    trimmed = data[start_sample:end_sample]

    trimmed_start_s = start_sample / sr
    trimmed_end_s = (len(data) - end_sample) / sr
    if trimmed_start_s > 0 or trimmed_end_s > 0:
        logger.info(
            "Trimmed silence: {:.3f}s from start, {:.3f}s from end (threshold={}dB)",
            trimmed_start_s, trimmed_end_s, threshold_db,
        )
    return trimmed


def _save_audio_to_disk(
    *,
    audio_path: str,
    dest: Path,
    sentence_index: int,
    text: str,
    userid: str,
    output_jsonl: Path,
    lock_path: Path,
) -> None:
    """Convert audio to mono 16 kHz WAV, trim silence, save, and append to output JSONL.

    Runs in a background thread so the UI stays responsive. After writing the
    WAV file, appends a NeMo-compatible JSONL entry under file lock.

    Parameters
    ----------
    audio_path : str
        Path to the raw recording from Gradio.
    dest : Path
        Destination WAV file path.
    sentence_index : int
        Index of the sentence in the input pool (for tracking).
    text : str
        The sentence text that was recorded.
    userid : str
        Hashed user identifier.
    output_jsonl : Path
        Path to append the completed entry to.
    lock_path : Path
        Path to the lock file for JSONL synchronization.
    """
    try:
        data, sr = sf.read(audio_path)
        logger.info("Read audio: shape={}, sr={}, duration={:.2f}s", data.shape, sr, len(data) / sr)
        if data.ndim > 1:
            data = data.mean(axis=1)
        target_sr = 16000
        if sr != target_sr:
            t_intp_start = time.time()
            target_len = int(len(data) / sr * target_sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            )
            t_intp_end = time.time()
            logger.info("Resampled {} Hz → {} Hz", sr, target_sr)
        # Trim leading/trailing silence for cleaner ASR training data.
        t_trim_start = time.time()
        data_before_trim = data
        data = _trim_silence(data=data, sr=target_sr)
        silence_trimmed = len(data) != len(data_before_trim)
        t_trim_end = time.time()
        sf.write(str(dest), data, target_sr, subtype="PCM_16")
        t_pcm_end = time.time()
        duration = len(data) / target_sr
        logger.info("Saved → {} (16 kHz mono, {:.2f}s)", dest, duration)

        dur_trim = t_trim_end - t_trim_start
        dur_wrt = t_pcm_end - t_trim_end
        if sr != target_sr:
            dur_intp = t_intp_end - t_intp_start
            logger.info("(interpolate: {}, trim: {}, write: {})", dur_intp, dur_trim, dur_wrt)
        else:
            logger.info("(trim: {}, write: {})", dur_trim, dur_wrt)

        # Append to output JSONL under lock.
        entry = {
            "audio_filepath": str(dest),
            "text": text,
            "duration": round(duration, 2),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "userid": userid,
            "sentence_index": sentence_index,
            "silence_trimmed": silence_trimmed,
        }
        with file_lock(lock_path):
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Appended JSONL entry for sentence {}", sentence_index)

    except Exception:
        logger.exception("Failed to save sentence {} to {}", sentence_index, dest)


# ── UI builder ────────────────────────────────────────────────────────────────


def build_ui(
    *,
    sentences: list[dict[str, Any]],
    output_dir: Path,
    output_jsonl: Path,
    users: list[tuple[str, str]],
    salt: str,
    lang: str | None = None,
) -> gr.Blocks:
    """Build the collaborative Gradio Blocks interface.

    Each authenticated user gets assigned a batch of sentences. Navigation
    cycles through the user's assigned batch. On save, the sentence is
    recorded and a replacement is drawn from the pool.

    Parameters
    ----------
    sentences : list[dict[str, Any]]
        Sentence entries from the input JSONL (each has ``"text"`` and
        optionally ``"audio_filepath"``).
    output_dir : Path
        Directory where WAV files are saved.
    output_jsonl : Path
        Path to the shared output JSONL file.
    users : list[tuple[str, str]]
        List of (username, password) for Gradio auth.
    salt : str
        Salt for userid hashing.

    Returns
    -------
    gr.Blocks
        The Gradio app, ready to ``.launch()``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_sentences = len(sentences)
    claims_path = output_dir / "claims.json"
    lock_path = output_dir / ".lock"

    # Per-user session state: {username: {"assigned": [indices], "cursor": int}}
    # Protected by a threading lock since Gradio callbacks run in threads.
    user_sessions: dict[str, dict[str, Any]] = {}
    sessions_lock = threading.Lock()

    def _get_username(request: gr.Request) -> str:
        """Extract username from Gradio request."""
        return request.username or "anonymous"

    def _get_or_create_session(username: str) -> dict[str, Any]:
        """Get or initialize the user's session, assigning sentences on first access."""
        with sessions_lock:
            if username not in user_sessions:
                assigned = assign_sentences(
                    username=username,
                    total_sentences=total_sentences,
                    claims_path=claims_path,
                    lock_path=lock_path,
                    output_jsonl=output_jsonl,
                )
                user_sessions[username] = {
                    "assigned": assigned,
                    "cursor": 0,
                    "completed": set(),  # in-memory cache of completed sentence indices
                }
                logger.info("Created session for '{}' with {} sentences", username, len(assigned))
            return user_sessions[username]

    def _display(username: str) -> tuple[str, str | None]:
        """Return the sentence HTML and optional reference audio path.

        The internal assignment/queue mechanism is hidden from the user;
        they only see the sentence to read aloud (or a completion message).
        When the input JSONL has an ``audio_filepath`` pointing to an
        existing file, it is returned so the UI can show a playback player
        (useful for validating TTS-generated datasets).

        Returns
        -------
        tuple[str, str | None]
            (sentence_html, reference_audio_path_or_None)
        """
        session = _get_or_create_session(username)
        assigned = session["assigned"]
        cursor = session["cursor"]
        if not assigned:
            return "<h2>No more sentences to record! 🎉</h2>", None
        idx = assigned[cursor]
        entry = sentences[idx]
        text = entry["text"]
        ref_audio = entry.get("audio_filepath")
        if ref_audio:
            logger.info("Serving reference audio for sentence {}: '{}'", idx, ref_audio)
        else:
            logger.info("No reference audio for sentence {}", idx)
        # Use gr.update() so the Audio component properly receives the value
        # and becomes visible/hidden accordingly.
        audio_update = gr.update(value=ref_audio, visible=ref_audio is not None)
        return f"<h2 style='line-height:1.6'>{text}</h2>", audio_update

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_load(request: gr.Request) -> tuple[str, str | None]:
        """Initialize the UI for the authenticated user on page load."""
        username = _get_username(request)
        logger.info("User '{}' loaded page", username)
        return _display(username)

    def flag_recording(*, nth_from_end: int = 1, request: gr.Request) -> str:
        """Flag the Nth most recent recording by this user (1 = last, 2 = second-to-last).

        Sets ``"flagged": true`` in the JSONL entry so it can be manually
        inspected later. This allows users to mark recordings they suspect
        may have issues without fully discarding them.

        Parameters
        ----------
        nth_from_end : int
            Which recording to flag, counted from the most recent (1-indexed).
        request : gr.Request
            Gradio request for user identification.

        Returns
        -------
        str
            Status message displayed to the user.
        """
        username = _get_username(request)
        userid = make_userid(username=username, salt=salt)

        with file_lock(lock_path):
            if not output_jsonl.exists():
                return "⚠️ Nothing to flag."

            lines = output_jsonl.read_text(encoding="utf-8").splitlines()

            # Find the Nth entry from the end belonging to this user.
            count_found = 0
            target_line_idx = None
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("userid") == userid:
                        count_found += 1
                        if count_found == nth_from_end:
                            target_line_idx = i
                            # Toggle: set flagged=true, or remove if already flagged.
                            if entry.get("flagged"):
                                del entry["flagged"]
                                lines[i] = json.dumps(entry, ensure_ascii=False)
                                output_jsonl.write_text(
                                    "\n".join(lines) + ("\n" if lines else ""),
                                    encoding="utf-8",
                                )
                                text_preview = entry["text"][:50]
                                logger.info("User '{}' unflagged recording for sentence {}", username, entry["sentence_index"])
                                return f"🏳️ Unflagged: \"{text_preview}{'…' if len(entry['text']) > 50 else ''}\""
                            else:
                                entry["flagged"] = True
                                lines[i] = json.dumps(entry, ensure_ascii=False)
                                output_jsonl.write_text(
                                    "\n".join(lines) + ("\n" if lines else ""),
                                    encoding="utf-8",
                                )
                                text_preview = entry["text"][:50]
                                logger.info("User '{}' flagged recording for sentence {}", username, entry["sentence_index"])
                                return f"🚩 Flagged: \"{text_preview}{'…' if len(entry['text']) > 50 else ''}\""
                except json.JSONDecodeError:
                    continue

            if target_line_idx is None:
                label = "last" if nth_from_end == 1 else f"#{nth_from_end} from last"
                return f"⚠️ No {label} recording found to flag."

        return "⚠️ Nothing to flag."

    def save_recording(audio_path: str | None, request: gr.Request) -> tuple[str, str | None, None]:
        """Save the current recording, mark sentence complete, assign a new one."""
        username = _get_username(request)
        session = _get_or_create_session(username)
        assigned = session["assigned"]

        if audio_path is None or not assigned:
            logger.warning("Save called with no recording or no sentences for '{}'", username)
            return *_display(username), None

        cursor = session["cursor"]
        sentence_index = assigned[cursor]
        text = sentences[sentence_index]["text"]
        userid = make_userid(username=username, salt=salt)

        # Generate unique filename: {userid}_{uuid4[:8]}.wav
        filename = f"{userid}_{uuid.uuid4().hex[:8]}.wav"
        dest = output_dir / filename

        logger.info("Saving sentence {} for user '{}' → {}", sentence_index, username, dest)

        # Fire off background save (audio conversion + JSONL append).
        threading.Thread(
            target=_save_audio_to_disk,
            kwargs=dict(
                audio_path=audio_path,
                dest=dest,
                sentence_index=sentence_index,
                text=text,
                userid=userid,
                output_jsonl=output_jsonl,
                lock_path=lock_path,
            ),
            daemon=True,
        ).start()
        logger.debug("Background save thread started for sentence {}", sentence_index)

        # Reassign: remove completed sentence, add a new one from the pool.
        # Pass in-memory completed cache to avoid re-reading the output JSONL.
        t_reassign = time.time()
        completed_cache = session.get("completed", set())
        new_idx = complete_and_reassign(
            username=username,
            sentence_index=sentence_index,
            total_sentences=total_sentences,
            claims_path=claims_path,
            lock_path=lock_path,
            output_jsonl=output_jsonl,
            completed_cache=completed_cache,
        )
        logger.info("complete_and_reassign took {:.3f}s for sentence {}", time.time() - t_reassign, sentence_index)

        with sessions_lock:
            session["completed"].add(sentence_index)
            assigned.remove(sentence_index)
            if new_idx is not None:
                assigned.append(new_idx)
            # Clamp cursor to valid range after removal.
            if assigned:
                session["cursor"] = session["cursor"] % len(assigned)
            else:
                session["cursor"] = 0

        return *_display(username), None

    def skip_sentence(request: gr.Request) -> tuple[str, str | None, None]:
        """Skip the current sentence without recording, releasing its claim.

        The skipped sentence's claim is removed so other users can pick it up.
        A new sentence is drawn from the pool to replace it.
        """
        username = _get_username(request)
        session = _get_or_create_session(username)
        assigned = session["assigned"]

        if not assigned:
            return *_display(username), None

        cursor = session["cursor"]
        sentence_index = assigned[cursor]

        logger.info("User '{}' skipping sentence {}", username, sentence_index)

        # Release the claim and get a replacement (reuses reassign logic,
        # but the sentence won't appear in the output JSONL since nothing was saved).
        new_idx = complete_and_reassign(
            username=username,
            sentence_index=sentence_index,
            total_sentences=total_sentences,
            claims_path=claims_path,
            lock_path=lock_path,
            output_jsonl=output_jsonl,
        )

        with sessions_lock:
            assigned.remove(sentence_index)
            if new_idx is not None:
                assigned.append(new_idx)
            if assigned:
                session["cursor"] = session["cursor"] % len(assigned)
            else:
                session["cursor"] = 0

        return *_display(username), None

    def on_recording_stop(
        audio_path: str | None, auto_advance: bool, request: gr.Request,
    ) -> tuple[str, str | None, None, gr.update]:
        """Handle recording stop: always save; advance to the next sentence only if auto_advance is True.

        When auto_advance is False, the saved recording is kept visible and a
        "Next" button appears so the user can manually move on.
        """
        t0 = time.time()
        username = _get_username(request)

        if audio_path is None:
            logger.warning("Recording stopped but Gradio returned None for '{}'", username)
            return *_display(username), None, gr.update(visible=False)

        p = Path(audio_path)
        file_size = p.stat().st_size if p.exists() else 0
        t_handler_start = time.time()
        logger.info(
            "Recording stopped for '{}' → {} ({} bytes) | handler invoked {:.3f}s after t0",
            username, audio_path, file_size, t_handler_start - t0,
        )

        # Capture current sentence text before reassignment (for frozen display).
        session = _get_or_create_session(username)
        assigned = session["assigned"]
        if assigned:
            current_text = sentences[assigned[session["cursor"]]]["text"]
        else:
            current_text = ""

        # Always save on recording stop.
        sentence_display, ref_audio_update, mic_clear = save_recording(audio_path=audio_path, request=request)
        logger.info("on_recording_stop total handler took {:.3f}s for '{}'", time.time() - t0, username)

        if auto_advance:
            return sentence_display, ref_audio_update, mic_clear, gr.update(visible=False)
        else:
            # Keep showing current sentence (now saved); reveal the Next button.
            frozen_sentence = f"<h2 style='line-height:1.6'>{current_text}</h2>"
            return frozen_sentence, gr.update(visible=False), mic_clear, gr.update(visible=True)

    def advance_to_next(request: gr.Request) -> tuple[str, str | None, None, gr.update]:
        """Manually advance to the next sentence (used when auto_advance is off)."""
        username = _get_username(request)
        sentence_display, ref_audio_update = _display(username)
        return sentence_display, ref_audio_update, None, gr.update(visible=False)

    # ── Custom JS for keyboard shortcuts ──────────────────────────────────
    shortcuts_js = """
    () => {
        if (window._audioRecorderShortcutsRegistered) return;
        window._audioRecorderShortcutsRegistered = true;

        document.addEventListener('keydown', (e) => {
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

            const mic = document.querySelector('#mic-input');
            if (!mic) return;

            if (e.code === 'Space') {
                e.preventDefault();
                const btn = mic.querySelector('button.record-button, button.stop-button');
                if (btn) btn.click();
            } else if (e.code === 'KeyR') {
                e.preventDefault();
                // Pause (not stop) to avoid triggering auto-save, then clear.
                const pauseBtn = mic.querySelector('button.pause-button');
                if (pauseBtn) pauseBtn.click();
                setTimeout(() => {
                    const clearBtn = mic.querySelector('button[aria-label="Clear"]');
                    if (clearBtn) clearBtn.click();
                }, 100);
            } else if (e.code === 'KeyS') {
                e.preventDefault();
                document.querySelector('#btn-skip')?.click();
            } else if (e.code === 'KeyF') {
                e.preventDefault();
                document.querySelector('#btn-flag')?.click();
            } else if (e.code === 'KeyG') {
                e.preventDefault();
                document.querySelector('#btn-flag-prev')?.click();
            }
        });
    }
    """

    # Restart JS: pause (not stop!) the recording, clear the mic, then re-record.
    # Using pause instead of stop avoids triggering the stop_recording event,
    # which would auto-save the partial recording. Pause just halts capture
    # so we can safely clear and start fresh.
    restart_js = """
    () => {
        const mic = document.querySelector('#mic-input');
        if (!mic) return;
        // Pause first to avoid triggering stop_recording (which auto-saves).
        const pauseBtn = mic.querySelector('button.pause-button');
        if (pauseBtn) pauseBtn.click();
        setTimeout(() => {
            const clearBtn = mic.querySelector('button[aria-label="Clear"]');
            if (clearBtn) clearBtn.click();
            setTimeout(() => {
                const recordBtn = mic.querySelector('button.record-button');
                if (recordBtn) recordBtn.click();
            }, 200);
        }, 100);
    }
    """

    # Auto-record JS: after saving, automatically start recording the next sentence
    # only if the auto-record checkbox is checked.
    auto_record_after_stop_js = """
    () => {
        const checkboxContainer = document.querySelector('#auto-record-checkbox');
        if (!checkboxContainer) return;
        const checkbox = checkboxContainer.querySelector('input[type="checkbox"]');
        if (!checkbox || !checkbox.checked) return;

        let attempts = 0;
        const poll = setInterval(() => {
            attempts++;
            const mic = document.querySelector('#mic-input');
            if (!mic) return;
            const recordBtn = mic.querySelector('button.record-button');
            if (recordBtn) {
                clearInterval(poll);
                recordBtn.click();
            } else if (attempts > 20) {
                clearInterval(poll);
            }
        }, 200);
    }
    """

    # ── Internationalization (i18n) ──────────────────────────────────────
    # French is the default language; English is also supported.
    # Gradio auto-detects browser locale, but we default to French.
    # When --lang is set, we duplicate the chosen language's translations
    # to all locales so the browser locale doesn't matter.
    translations: dict[str, dict[str, str]] = {}
    translations["en"] = {
            "app_title": "# 🎙️ AudioCrowd\n### Collaborative Audio Dataset Recorder",
            "about_label": "About this project",
            "about_text": (
                "This tool collects human voice recordings for fine-tuning an ASR "
                "(Automatic Speech Recognition) model on medical sentences. "
                "Simply read the displayed sentence aloud, and your recording "
                "will be saved automatically.\n\n"
                "**Keyboard shortcuts:** Space = record/stop, R = reset, "
                "S = skip, F = flag current sample, G = flag previous sample\n\n"
                "*Built with [Claude Code](https://claude.com/claude-code).*"
            ),
            "instructions_label": "How to record properly",
            "instructions_text": (
                "Speak as you would to a dictation device — usually that means **quite fast**, "
                "with a natural flow.\n\n"
                "- **If you mispronounce a word, you must start over** (use the Restart button "
                "or press R, then re-record).\n"
                "- Leading and trailing silences are automatically removed.\n"
                "- **Ambient noise is fine** and actually helps model robustness.\n"
                "- **Respect the punctuation** as much as possible (pauses for commas, "
                "intonation for questions, etc.).\n"
                "- **Hesitations are acceptable** (uh, um…), but mispronunciations are not.\n"
                "- **Flagging:** If you suspect a recording has an issue but don't want to discard it, "
                "press F to flag the current sample or G to flag the previous one. "
                "Flagged recordings are kept but marked for manual review. Press again to unflag."
            ),
            "mic_label": "Your recording (Space to toggle, R to reset)",
            "btn_skip": "⏭️ Skip",
            "btn_restart": "🔄 Restart (R)",
            "btn_flag": "🚩 Flag current sample (F)",
            "btn_flag_prev": "🚩 Flag previous sample (G)",
            "ref_audio_label": "Reference audio (original)",
            "auto_record_label": "Auto-advance to next sentence after each save",
            "btn_next": "Next sentence ▶",
    }
    translations["fr"] = {
            "app_title": "# 🎙️ AudioCrowd\n### Enregistreur Collaboratif de Jeux de Données Audio",
            "about_label": "À propos de ce projet",
            "about_text": (
                "Cet outil collecte des enregistrements vocaux pour affiner un modèle "
                "de reconnaissance automatique de la parole (ASR) sur des phrases médicales. "
                "Lisez simplement la phrase affichée à voix haute, et votre enregistrement "
                "sera sauvegardé automatiquement.\n\n"
                "**Raccourcis clavier :** Espace = enregistrer/arrêter, R = réinitialiser, "
                "S = passer, F = signaler l'actuel, G = signaler le précédent\n\n"
                "*Construit avec [Claude Code](https://claude.com/claude-code).*"
            ),
            "instructions_label": "Comment bien enregistrer",
            "instructions_text": (
                "Parlez comme si vous dictiez à un appareil de reconnaissance vocale — "
                "c'est-à-dire généralement **assez vite**, avec un débit naturel.\n\n"
                "- **Si vous prononcez mal un mot, vous devez recommencer** (utilisez le bouton "
                "Recommencer ou appuyez sur R, puis ré-enregistrez).\n"
                "- Les silences en début et fin d'enregistrement sont automatiquement supprimés.\n"
                "- **Le bruit ambiant est acceptable** et aide même à la robustesse du modèle.\n"
                "- **Respectez la ponctuation** autant que possible (pauses pour les virgules, "
                "intonation pour les questions, etc.).\n"
                "- **Les hésitations sont acceptables** (euh, hum…), mais pas les erreurs "
                "de prononciation.\n"
                "- **Signalement :** Si vous pensez qu'un enregistrement a un problème sans vouloir "
                "le supprimer, appuyez sur F pour signaler l'échantillon actuel ou G pour le précédent. "
                "Les enregistrements signalés sont conservés mais marqués pour vérification manuelle. "
                "Appuyez à nouveau pour retirer le signalement."
            ),
            "mic_label": "Votre enregistrement (Espace pour démarrer/arrêter, R pour réinitialiser)",
            "btn_skip": "⏭️ Passer",
            "btn_restart": "🔄 Recommencer (R)",
            "btn_flag": "🚩 Signaler l'échantillon actuel (F)",
            "btn_flag_prev": "🚩 Signaler l'échantillon précédent (G)",
            "ref_audio_label": "Audio de référence (original)",
            "auto_record_label": "Passer automatiquement à la phrase suivante après chaque sauvegarde",
            "btn_next": "Phrase suivante ▶",
    }

    # When --lang is set, force that language for all locales
    if lang is not None:
        forced = translations[lang]
        for key in translations:
            translations[key] = forced

    i18n = gr.I18n(**translations)

    # ── Layout ────────────────────────────────────────────────────────────

    # default_concurrency_limit controls how many concurrent callback executions
    # Gradio allows per event handler. Setting it higher allows multiple users
    # to interact simultaneously without blocking each other's requests.
    with gr.Blocks(
        title="AudioCrowd",
    ) as demo:
        gr.Markdown(i18n("app_title"))

        # Collapsible info banner explaining the project.
        with gr.Accordion(label=i18n("about_label"), open=False):
            gr.Markdown(i18n("about_text"))

        # Collapsible recording instructions for volunteers.
        with gr.Accordion(label=i18n("instructions_label"), open=False):
            gr.Markdown(i18n("instructions_text"))

        sentence = gr.Markdown(value="")

        # Reference audio player: shown only when the input JSONL contains
        # an audio_filepath pointing to an existing file (e.g. TTS-generated
        # datasets). Allows the user to listen to the original audio while
        # re-recording it with their own voice.
        ref_audio = gr.Audio(
            label=i18n("ref_audio_label"),
            type="filepath",
            interactive=False,
            elem_id="ref-audio",
        )

        mic = gr.Audio(
            label=i18n("mic_label"),
            sources=["microphone"],
            type="filepath",
            elem_id="mic-input",
            format="wav",
            editable=False,
            waveform_options=gr.WaveformOptions(
                show_recording_waveform=False,
            ),
        )

        with gr.Row():
            skip_btn = gr.Button(i18n("btn_skip"), elem_id="btn-skip")
            restart_btn = gr.Button(i18n("btn_restart"), elem_id="btn-restart")

        with gr.Row():
            flag_prev_btn = gr.Button(i18n("btn_flag_prev"), elem_id="btn-flag-prev")
            flag_btn = gr.Button(i18n("btn_flag"), elem_id="btn-flag")

        auto_record_cb = gr.Checkbox(
            label=i18n("auto_record_label"),
            value=False,
            elem_id="auto-record-checkbox",
        )

        next_btn = gr.Button(i18n("btn_next"), elem_id="btn-next", visible=False)

        # Status line for flag feedback messages.
        action_status = gr.Markdown(value="")

        # ── Wire up events ────────────────────────────────────────────────

        # On page load: initialize the user's session and display first sentence.
        demo.load(
            fn=on_load,
            inputs=None,
            outputs=[sentence, ref_audio],
        )

        outputs = [sentence, ref_audio, mic]
        outputs_with_next = [sentence, ref_audio, mic, next_btn]

        # Skip: release current sentence claim and move to next.
        skip_btn.click(
            fn=skip_sentence,
            inputs=None,
            outputs=outputs,
        )

        # Restart: stop current recording, clear audio, and start recording again.
        # Pure client-side action — no server callback needed, just JS.
        restart_btn.click(fn=None, inputs=None, outputs=None, js=restart_js)

        # Flag: toggle flagged status on the last or second-to-last recording.
        def _flag_last(request: gr.Request) -> str:
            return flag_recording(nth_from_end=1, request=request)

        def _flag_prev(request: gr.Request) -> str:
            return flag_recording(nth_from_end=2, request=request)

        flag_btn.click(
            fn=_flag_last,
            inputs=None,
            outputs=[action_status],
        )
        flag_prev_btn.click(
            fn=_flag_prev,
            inputs=None,
            outputs=[action_status],
        )

        # Recording auto-saves on stop. Advances automatically only if checkbox is checked.
        # If the user mispronounced, they can flag the recording.
        mic.stop_recording(
            fn=on_recording_stop,
            inputs=[mic, auto_record_cb],
            outputs=outputs_with_next,
        ).then(fn=None, js=auto_record_after_stop_js)

        # Manual advance when auto-advance is off.
        next_btn.click(
            fn=advance_to_next,
            inputs=None,
            outputs=outputs_with_next,
        )

        # Register keyboard shortcuts.
        demo.load(fn=None, inputs=None, outputs=None, js=shortcuts_js)

    return demo, i18n


# ── CLI entry point ───────────────────────────────────────────────────────────


@click.command()
@click.argument("jsonl_path", type=click.Path(exists=True), envvar="JSONL_PATH")
@click.option(
    "--users-csv",
    required=True,
    type=click.Path(exists=True),
    envvar="USERS_CSV",
    help="CSV file with 'username,password' rows (no header).",
)
@click.option(
    "--output-dir",
    default="./recordings/",
    show_default=True,
    envvar="OUTPUT_DIR",
    help="Directory to save WAV recordings into.",
)
@click.option(
    "--output-jsonl",
    default="./output.jsonl",
    show_default=True,
    envvar="OUTPUT_JSONL",
    help="Output JSONL file for completed recordings.",
)
@click.option(
    "--salt",
    default="audiorec",
    show_default=True,
    envvar="SALT",
    help="Fixed salt for deterministic userid hashing.",
)
@click.option(
    "--port",
    default=7860,
    show_default=True,
    type=int,
    envvar="PORT",
    help="Port to serve the Gradio UI on.",
)
@click.option(
    "--share",
    is_flag=True,
    default=False,
    envvar="SHARE",
    help="Create a public Gradio share link.",
)
@click.option(
    "--lang",
    default=None,
    type=click.Choice(["en", "fr"], case_sensitive=False),
    envvar="LANG_OVERRIDE",
    help="Force UI language instead of auto-detecting from browser.",
)
def main(
    jsonl_path: str,
    users_csv: str,
    output_dir: str,
    output_jsonl: str,
    salt: str,
    port: int,
    share: bool,
    lang: str | None,
) -> None:
    """Collaborative multi-user audio dataset recorder for ASR fine-tuning.

    JSONL_PATH is a file where each line has at least a ``text`` field.
    """
    users = load_users_csv(csv_path=users_csv)
    if not users:
        raise click.ClickException("No users found in the CSV file.")

    sentences = load_sentences(jsonl_path=jsonl_path)
    if not sentences:
        raise click.ClickException("No sentences found in the JSONL file.")

    # Collect unique directories containing reference audio files so Gradio
    # can serve them (it restricts file access by default for security).
    ref_audio_dirs: set[str] = set()
    for s in sentences:
        if "audio_filepath" in s:
            ref_audio_dirs.add(str(Path(s["audio_filepath"]).parent))
    if ref_audio_dirs:
        logger.info("Reference audio found in {} directories, enabling Gradio access", len(ref_audio_dirs))

    output_dir_path = Path(output_dir)
    output_jsonl_path = Path(output_jsonl)

    demo, i18n = build_ui(
        sentences=sentences,
        output_dir=output_dir_path,
        output_jsonl=output_jsonl_path,
        users=users,
        salt=salt,
        lang=lang,
    )
    # Enable the request queue for proper concurrent user handling, and set
    # default_concurrency_limit high enough for multiple simultaneous users.
    # max_size=100 caps the queue to prevent memory issues under heavy load.
    demo.queue(default_concurrency_limit=40, max_size=100)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=False,
        share=share,
        auth=users,
        i18n=i18n,
        allowed_paths=list(ref_audio_dirs) if ref_audio_dirs else None,
    )


if __name__ == "__main__":
    main()
