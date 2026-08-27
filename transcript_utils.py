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


def _parse_vtt(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or _CUE_NUMBER_RE.match(line) or _TIMESTAMP_RE.match(line):
            continue
        m = _VOICE_TAG_RE.search(line)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(line)
    return "\n".join(lines)


def load_transcript(path: str) -> str:
    """Returns (text, error). Exactly one is None."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as exc:
        return None, f"Couldn't read {path}: {exc}"

    if path.lower().endswith(".vtt"):
        text = _parse_vtt(raw)
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
