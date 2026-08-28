"""Transcript loading and chunking for the MoM/action-items agent.

Phase 1 reads a manually-provided transcript file (e.g. exported from Teams
as .vtt, or plain .txt). Phase 2 (once tenant admin grants Graph transcript
access -- see task_breakdown_chat.py's mom command docstring) will fetch the
same shape of plain text directly from Teams instead; nothing downstream of
load_transcript() needs to change for that.
"""

import os
import re

# Teams' actual .vtt export looks like:
#   WEBVTT
#
#   1
#   00:00:01.000 --> 00:00:04.000
#   <v Jane Doe>Let's get started with the sprint review.</v>
#
# Cue number and timestamp lines are stripped; <v Speaker>text</v> becomes a
# plain "Speaker: text" line -- more compact and gives the model clean
# speaker attribution for action-item ownership.
_CUE_NUMBER_RE = re.compile(r"^\d+$")
_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")
_VOICE_TAG_RE = re.compile(r"<v\s+([^>]+)>(.*?)</v>", re.DOTALL)


def _parse_vtt(raw: str, keep_timestamps: bool = False) -> str:
    """keep_timestamps=False (the /mom default, unchanged) drops cue
    numbers and timing entirely -- MoM never needed them. keep_timestamps=
    True (for /requirements, which needs a real citation trail) prefixes
    each speaker line with its cue's start time instead of discarding it,
    e.g. "[00:00:03.500] Jane Doe: Let's get started" -- read from the
    timing line that precedes each <v> cue, since the cue itself doesn't
    carry it."""
    lines = []
    pending_start = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or _CUE_NUMBER_RE.match(line):
            continue
        if _TIMESTAMP_RE.match(line):
            pending_start = line.split("-->")[0].strip()
            continue
        m = _VOICE_TAG_RE.search(line)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            prefix = f"[{pending_start}] " if keep_timestamps and pending_start else ""
            lines.append(f"{prefix}{speaker}: {text}")
        else:
            lines.append(line)
    return "\n".join(lines)


def load_transcript(path: str, keep_timestamps: bool = False) -> str:
    """Returns (text, error). Exactly one is None. keep_timestamps only
    affects .vtt input -- see _parse_vtt(); plain .txt has no timing
    information to preserve either way."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as exc:
        return None, f"Couldn't read {path}: {exc}"

    if path.lower().endswith(".vtt"):
        text = _parse_vtt(raw, keep_timestamps=keep_timestamps)
    else:
        text = raw.strip()

    if not text:
        return None, f"{path} appears to be empty after parsing."
    return text, None


def chunk_transcript(text: str, max_chars: int = 6000) -> list:
    """Splits on line boundaries (speaker turns) so a chunk never cuts a
    sentence mid-way -- an hour-long meeting can easily be 8,000-15,000+
    words, well past what fits in one context window alongside the rest of
    the prompt, so this is the normal path for a real transcript, not just
    an edge case."""
    lines = text.splitlines()
    chunks = []
    current = []
    current_len = 0
    for line in lines:
        # +1 for the newline that will rejoin this line to the chunk.
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]
