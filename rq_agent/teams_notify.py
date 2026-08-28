"""Posting the generated MoM back into a Teams channel via an Incoming
Webhook / "Workflows" connector -- the low-friction alternative to the
Graph API's ChannelMessage.Send, which is delegated-only (no app-only,
admin-consented way to post as a background service the way transcript
reads might eventually get -- see task_breakdown_chat.py's mom command
docstring for that side of things).

Setup (one-time, no tenant admin needed beyond the channel allowing
connectors, which is usually on by default):
  1. In the target Teams channel: "..." -> Workflows -> "Post to a channel
     when a webhook request is received" -> name it, finish the wizard.
  2. Copy the webhook URL it gives you.
  3. export TEAMS_WEBHOOK_URL="https://...".contoso.com/workflows/..."
"""

import json
import urllib.error
import urllib.request


def _is_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _clean_heading(line: str) -> str:
    return line.lstrip("#").strip()


def _clean_bullet(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("- [ ]"):
        return "☐ " + stripped[len("- [ ]"):].strip()  # ☐
    if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
        return "☑ " + stripped[len("- [x]"):].strip()  # ☑
    if stripped.startswith("- "):
        return "• " + stripped[2:].strip()  # •
    return stripped


def _build_adaptive_card(title: str, markdown_text: str) -> dict:
    """One TextBlock per non-empty line rather than relying on the host's
    markdown-newline handling inside a single TextBlock (inconsistent across
    Teams clients) -- this guarantees each line lands on its own row."""
    body = [{
        "type": "TextBlock",
        "text": title,
        "weight": "bolder",
        "size": "large",
        "wrap": True,
    }]
    for line in markdown_text.splitlines():
        if not line.strip():
            continue
        if _is_heading(line):
            body.append({
                "type": "TextBlock",
                "text": _clean_heading(line),
                "weight": "bolder",
                "size": "medium",
                "spacing": "medium",
                "wrap": True,
            })
        else:
            body.append({
                "type": "TextBlock",
                "text": _clean_bullet(line),
                "wrap": True,
            })

    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": body,
            },
        }],
    }


def send_to_teams(webhook_url: str, title: str, markdown_text: str):
    """POSTs the MoM to the channel's webhook. Returns (True, None) on
    success, (False, error_message) otherwise -- same tuple convention as
    transcript_utils.load_transcript()."""
    payload = _build_adaptive_card(title, markdown_text)
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if 200 <= resp.status < 300:
                return True, None
            return False, f"Teams webhook returned HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return False, f"Teams webhook returned HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        return False, f"Couldn't reach Teams webhook: {exc.reason}"
