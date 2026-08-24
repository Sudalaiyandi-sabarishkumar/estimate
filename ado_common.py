"""Shared Azure DevOps + Ollama plumbing for the task-breakdown agent.

Used by both azure_task_breakdown.py (one-shot) and task_breakdown_chat.py
(interactive). Handles:
  - fetching a work item plus its linked items, attachments, and comments
  - pulling real estimate-vs-actual examples from past closed Tasks, to use
    as few-shot calibration for the model
  - persisting user corrections locally so future sessions on this machine
    reuse them (see CORRECTIONS_FILE)
"""

import base64
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

DEFAULT_ORG = "rootquotient"
DEFAULT_PROJECT = "Buddhi Mantra"
OLLAMA_URL = "http://localhost:11434/api/chat"
CORRECTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrections.json")
DEFAULT_MODEL = "qwen2.5-coder:7b"

LINK_LABELS = {
    "System.LinkTypes.Hierarchy-Forward": "Child",
    "System.LinkTypes.Hierarchy-Reverse": "Parent",
    "System.LinkTypes.Related": "Related",
    "System.LinkTypes.Dependency-Forward": "Depends on",
    "System.LinkTypes.Dependency-Reverse": "Blocks",
}


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _auth_header(pat: str) -> dict:
    return {"Authorization": f"Basic {base64.b64encode(f':{pat}'.encode()).decode()}"}


def _get(url: str, pat: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=_auth_header(pat))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_work_item(org: str, project: str, item_id: str, pat: str):
    """Returns (work_item, error_message). Exactly one of the two is None."""
    url = (
        f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
        f"/_apis/wit/workitems/{item_id}?$expand=all&api-version=7.1"
    )
    try:
        return _get(url, pat), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            error = (f"Work item #{item_id} wasn't found in the '{project}' project "
                      f"(it may exist in a different project, or not exist at all).")
        else:
            error = f"Azure DevOps API error {exc.code}: {body}"
        return None, error
    except urllib.error.URLError as exc:
        return None, (f"Network error reaching Azure DevOps ({exc.reason}). "
                       f"This is usually transient — try the same load again.")
    except Exception as exc:
        return None, f"Couldn't fetch work item: {exc}"


def fetch_related_context(org: str, project: str, work_item: dict, pat: str) -> str:
    """Linked work items, attachment filenames, and recent comments — cheap
    extra grounding beyond the item's own title/description."""
    relations = work_item.get("relations", []) or []
    related_ids = []
    attachments = []
    for r in relations:
        rel = r.get("rel", "")
        if rel == "AttachedFile":
            name = r.get("attributes", {}).get("name", "attachment")
            attachments.append(name)
        elif rel in LINK_LABELS:
            m = re.search(r"/workItems/(\d+)$", r.get("url", ""))
            if m:
                related_ids.append((LINK_LABELS[rel], m.group(1)))

    lines = []

    if related_ids:
        lines.append("Linked work items:")
        for label, rid in related_ids[:5]:
            try:
                url = (
                    f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
                    f"/_apis/wit/workitems/{rid}?fields=System.Title,System.WorkItemType,System.State&api-version=7.1"
                )
                item = _get(url, pat)
                f = item.get("fields", {})
                lines.append(f"  - {label}: #{rid} [{f.get('System.WorkItemType', '?')}, "
                              f"{f.get('System.State', '?')}] {f.get('System.Title', '')}")
            except Exception:
                lines.append(f"  - {label}: #{rid} (couldn't fetch title)")

    if attachments:
        lines.append("Attachments (not parsed — check these manually for design details): "
                      + ", ".join(attachments[:10]))

    item_id = work_item.get("id")
    try:
        comments_url = (
            f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
            f"/_apis/wit/workItems/{item_id}/comments?api-version=7.1-preview.3"
        )
        comments = _get(comments_url, pat).get("comments", [])
        if comments:
            lines.append("Recent comments:")
            for c in comments[-3:]:
                text = strip_html(c.get("text", ""))[:300]
                if text:
                    lines.append(f'  - "{text}"')
    except Exception:
        pass

    return "\n".join(lines)


def build_context_message(work_item: dict, item_id: str, related_context: str = "") -> str:
    fields = work_item.get("fields", {})
    title = fields.get("System.Title", "")
    item_type = fields.get("System.WorkItemType", "")
    description = strip_html(fields.get("System.Description", ""))
    acceptance = strip_html(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", ""))
    repro_steps = strip_html(fields.get("Microsoft.VSTS.TCM.ReproSteps", ""))

    parts = [f"WORK ITEM CONTEXT (#{item_id})", f"Type: {item_type}", f"Title: {title}"]
    if description:
        parts.append(f"Description:\n{description}")
    if repro_steps:
        parts.append(f"Details:\n{repro_steps}")
    if acceptance:
        parts.append(f"Acceptance criteria:\n{acceptance}")
    if related_context:
        parts.append(related_context)
    return "\n\n".join(parts)


def fetch_reference_examples(org: str, project: str, pat: str, count: int = 20) -> str:
    """Pull recent closed Tasks with real OriginalEstimate/CompletedWork hours,
    to use as few-shot calibration for how this team actually estimates."""
    try:
        wiql_url = (
            f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
            f"/_apis/wit/wiql?api-version=7.1"
        )
        query = {
            "query": (
                "SELECT [System.Id] FROM WorkItems "
                "WHERE [System.TeamProject] = @project "
                "AND [System.WorkItemType] = 'Task' "
                "AND [System.State] IN ('Closed', 'Done') "
                "ORDER BY [System.ChangedDate] DESC"
            )
        }
        req = urllib.request.Request(
            wiql_url,
            data=json.dumps(query).encode("utf-8"),
            headers={**_auth_header(pat), "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            ids = [str(w["id"]) for w in json.loads(resp.read().decode("utf-8")).get("workItems", [])][:60]
        if not ids:
            return ""

        fields = "System.Title,Microsoft.VSTS.Scheduling.OriginalEstimate,Microsoft.VSTS.Scheduling.CompletedWork"
        batch_url = (
            f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
            f"/_apis/wit/workitems?ids={','.join(ids)}&fields={fields}&api-version=7.1"
        )
        items = _get(batch_url, pat).get("value", [])

        examples = []
        for item in items:
            f = item.get("fields", {})
            est = f.get("Microsoft.VSTS.Scheduling.OriginalEstimate")
            actual = f.get("Microsoft.VSTS.Scheduling.CompletedWork")
            title = f.get("System.Title", "")
            if title and (est or actual):
                examples.append((title, est, actual))
            if len(examples) >= count:
                break

        if not examples:
            return ""

        lines = ["REFERENCE EXAMPLES FROM THIS TEAM'S PAST COMPLETED WORK "
                 "(use these to calibrate your estimates — note how actual can differ from estimated):"]
        for title, est, actual in examples:
            est_str = f"{est}h" if est is not None else "n/a"
            actual_str = f"{actual}h" if actual is not None else "n/a"
            lines.append(f'  - "{title}" — estimated: {est_str}, actual: {actual_str}')
        return "\n".join(lines)
    except Exception:
        return ""


def load_corrections() -> list:
    """Corrections saved on this machine from past sessions (see save_correction)."""
    if not os.path.exists(CORRECTIONS_FILE):
        return []
    try:
        with open(CORRECTIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_correction(item_id, note: str) -> None:
    corrections = load_corrections()
    corrections.append({
        "id": item_id,
        "note": note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)


def build_corrections_context(corrections: list, limit: int = 20) -> str:
    if not corrections:
        return ""
    lines = ["CORRECTIONS YOU'VE GIVEN IN PAST SESSIONS ON THIS MACHINE "
             "(apply these lessons where relevant):"]
    for c in corrections[-limit:]:
        tag = f"#{c['id']}" if c.get("id") else "general"
        lines.append(f"  - [{tag}] {c['note']}")
    return "\n".join(lines)


def corrections_for_item(corrections: list, item_id: str) -> list:
    return [c["note"] for c in corrections if c.get("id") == item_id]


def build_item_corrections_block(notes: list) -> str:
    """Strong, item-scoped, mechanical override — meant to sit right next to a
    loaded item's WORK ITEM CONTEXT so a small model can't deprioritize it the
    way it can a generic note buried in the system prompt, and can't satisfy
    it by merely citing it instead of actually changing the breakdown."""
    if not notes:
        return ""
    lines = ["CORRECTIONS FOR THIS SPECIFIC ITEM — mandatory, not optional:"]
    lines += [f"  - {n}" for n in notes]
    lines.append(
        "For each correction above, actually change the breakdown to match it: "
        "add, remove, or resize real tasks with real hour estimates. Do not just "
        "mention or cite the correction in a task description. Never write \"N/A\" "
        "or \"not needed\" for something a correction says is required. If a "
        "correction says backend work is needed, give Backend its own tasks and "
        "hours in the Backend section — do not fold it into a Frontend sub-bullet."
    )
    return "\n".join(lines)


def call_ollama(messages: list, model: str = DEFAULT_MODEL, show_progress: bool = True) -> str:
    """Streams the response from Ollama. When show_progress is True, tokens are
    printed to stdout as they arrive so a slow local generation doesn't look
    like a hang; the full text is still returned at the end either way."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
            "num_predict": 1200,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            chunks = []
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                piece = json.loads(line)
                content = piece.get("message", {}).get("content", "")
                if content:
                    if show_progress:
                        print(content, end="", flush=True)
                    chunks.append(content)
                if piece.get("done"):
                    break
    except urllib.error.URLError as exc:
        print(f"Could not reach Ollama at {OLLAMA_URL} ({exc}). Is it running? Try: brew services start ollama")
        return None
    if show_progress:
        print()
    return "".join(chunks).strip()
