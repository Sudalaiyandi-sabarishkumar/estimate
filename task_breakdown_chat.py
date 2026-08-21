#!/usr/bin/env python3
"""Interactive, conversational task-breakdown agent grounded in Azure DevOps.

Start a session, load a work item, then keep asking follow-up questions —
"just frontend", "what's risky here", "re-estimate without the design being
ready" — and the model answers from the same loaded context, remembering the
conversation like a chat.

Each loaded work item is grounded with its linked items, attachments, and
recent comments. The session also seeds itself once with real
estimate-vs-actual examples from this team's past closed Tasks, so the model
calibrates against how this team actually estimates rather than guessing.

Setup (one-time):
  export AZURE_DEVOPS_PAT="..."   # PAT with Work Items (Read) scope

Usage:
  python3 task_breakdown_chat.py
  python3 task_breakdown_chat.py 97061   # load a work item immediately
"""

import re
import sys
import os

from ado_common import (
    DEFAULT_ORG,
    DEFAULT_PROJECT,
    DEFAULT_MODEL,
    fetch_work_item,
    fetch_related_context,
    fetch_reference_examples,
    build_context_message,
    call_ollama,
)

BASE_SYSTEM_PROMPT = """You are a senior engineering lead helping a developer plan and \
estimate real Azure DevOps work items, in a back-and-forth conversation.

When work item context is given to you (marked "WORK ITEM CONTEXT"), treat it as \
ground truth and base everything on it — do not invent requirements it doesn't \
contain. Linked work items, attachments, and comments are extra context — use them, \
and mention when an attachment (e.g. a design file) likely has details you can't see.

Answer only what is asked:
- A request for a full breakdown -> give tasks grouped by Frontend, Backend, \
Testing, and Other, each with an hour estimate (S = under 4h, M = 4-16h, \
L = over 16h) for one mid-level engineer, plus a total.
- A request for just one category (e.g. "just frontend") -> give only that \
category's tasks and subtotal, don't repeat the others.
- A question about risk, scope, dependencies, or assumptions -> answer directly \
and briefly, no need to re-list every task.
- A yes/no or clarifying question about something you already said (e.g. "so no \
backend task?", "is that right?") -> answer the question directly in 1-2 sentences \
("No, this work item is frontend-only — no backend logic was needed."). NEVER \
respond to this kind of question by reprinting the full breakdown again.
- A request to re-estimate under a different assumption -> redo the estimate \
using that assumption, and say what changed.

If no work item context has ever been loaded and the question depends on one, say \
so and suggest "load <work-item-id>" rather than guessing.

Only trust work item details that were given to you in a WORK ITEM CONTEXT block. \
If the user asks about a work item ID you were never given a WORK ITEM CONTEXT \
block for, say you don't have it loaded and suggest "load <that id>" — never reuse \
another item's title or description for it, even if it "looks similar."

Keep responses concise and skip filler like "Sure, here's..." — answer directly.
"""

ID_RE = re.compile(r"\b\d{4,}\b")


def load_item(messages: list, user_input: str, pat: str, loaded_ids: set):
    match = ID_RE.search(user_input)
    if not match:
        print(f"Couldn't find a work item number in that. Try: load 97061\n")
        return
    item_id = match.group()
    print(f"Loading work item #{item_id} from Azure DevOps...")
    work_item, error = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, item_id, pat)
    if work_item is None:
        print(error + "\n")
        # Put the failure in the conversation so a follow-up "why did that fail?"
        # gets a real answer instead of a generic "I have no tool access" reply.
        messages.append({"role": "assistant",
                          "content": f"(Attempted to load work item #{item_id} but it failed: {error})"})
        return
    loaded_ids.add(item_id)
    related = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, work_item, pat)
    context = build_context_message(work_item, item_id, related)

    # Whatever the user asked for beyond "load <id>" becomes the actual request,
    # e.g. "load 97061 and give only the title" -> "and give only the title".
    instruction = re.sub(r"\bload\b", "", user_input, flags=re.IGNORECASE)
    instruction = re.sub(r"\bwork\s*items?\b", "", instruction, flags=re.IGNORECASE)
    instruction = instruction.replace(item_id, "")
    instruction = re.sub(r"\s+", " ", instruction).strip(" ,.")
    if not instruction:
        instruction = "Give me a full breakdown."

    messages.append({"role": "user", "content": context + "\n\n" + instruction})
    print()
    reply = call_ollama(messages, DEFAULT_MODEL)
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    print()


def main():
    pat = os.environ.get("AZURE_DEVOPS_PAT")
    if not pat:
        sys.exit("AZURE_DEVOPS_PAT is not set. Run: export AZURE_DEVOPS_PAT=\"...\"")

    system_prompt = BASE_SYSTEM_PROMPT
    examples = fetch_reference_examples(DEFAULT_ORG, DEFAULT_PROJECT, pat)
    if examples:
        system_prompt += "\n\n" + examples

    messages = [{"role": "system", "content": system_prompt}]
    loaded_ids = set()

    print("task-breakdown chat. Commands: 'load <id>' to load/switch a work item, 'exit' to quit.")
    if len(sys.argv) > 1:
        load_item(messages, sys.argv[1], pat, loaded_ids)
    else:
        print("No work item loaded yet. Type 'load <id>' or ask a general question.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            # Treat it as a load request whenever "load" and a work item number
            # appear together anywhere in the message, not just as a prefix —
            # covers phrasing like "now load workitem 97061 and give the title".
            if re.search(r"\bload\b", user_input, re.IGNORECASE) and ID_RE.search(user_input):
                load_item(messages, user_input, pat, loaded_ids)
                continue

            # Warn the model off reusing another item's data for an ID it was
            # never actually given context for.
            mentioned_ids = set(ID_RE.findall(user_input))
            unknown_ids = mentioned_ids - loaded_ids
            content = user_input
            if unknown_ids:
                known = ", ".join(f"#{i}" for i in loaded_ids) or "none"
                unknown = ", ".join(f"#{i}" for i in unknown_ids)
                content += (
                    f"\n\n(System note: you only have real WORK ITEM CONTEXT for {known}. "
                    f"You were never given context for {unknown} — do not reuse another "
                    f"item's details for it. Say you don't have it and suggest 'load {list(unknown_ids)[0]}'.)"
                )

            messages.append({"role": "user", "content": content})
            reply = call_ollama(messages, DEFAULT_MODEL)
            if reply is not None:
                messages.append({"role": "assistant", "content": reply})
            print()
        except Exception as exc:
            print(f"Something went wrong, but the session is still up: {exc}\n")


if __name__ == "__main__":
    main()
