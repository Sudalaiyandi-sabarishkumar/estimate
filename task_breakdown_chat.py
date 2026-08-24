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

Say "correct <note>" any time to save a correction (tied to the current work
item if one is loaded). It's saved to corrections.json on this machine and
folded into the system prompt of every future session, so the model keeps
the lesson going forward — it does not sync to other machines.

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
    load_corrections,
    save_correction,
    build_corrections_context,
    corrections_for_item,
    build_item_corrections_block,
)

BASE_SYSTEM_PROMPT = """You are a senior engineering lead helping a developer plan and \
estimate real Azure DevOps work items, in a back-and-forth conversation.

When work item context is given to you (marked "WORK ITEM CONTEXT"), treat it as \
ground truth and base everything on it — do not invent requirements it doesn't \
contain. Linked work items, attachments, and comments are extra context — use them, \
and mention when an attachment (e.g. a design file) likely has details you can't see. \
The one exception: a "CORRECTIONS FOR THIS SPECIFIC ITEM" block, when present, comes \
from the actual developer's real-world knowledge of the work and OVERRIDES the ticket \
text wherever they conflict — e.g. if a correction says backend work is needed, include \
it even if the ticket's acceptance criteria only describe UI behavior.

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

When asked to analyse the dependency between two work items, use this method (worked \
example):

  Item A: "Splash screen: on launch, show the logo for up to 3 seconds, then navigate \
  to the 'Get started' page."
  Item B: "Create account: when the user clicks 'Get started', navigate to the sign-up \
  form, then verify their mobile number via OTP."

  Step 1 — find the shared point: both items mention the "Get started" page.
  Step 2 — determine direction with this exact rule: whichever item's flow ENDS at the \
  shared point is the one being DEPENDED ON (it must exist first). Whichever item's \
  flow STARTS at the shared point is the DEPENDENT one (it needs the other to exist \
  first). Do not guess the direction from which item "sounds more important" or which \
  was mentioned first — only from which one ends there vs. starts there.
  Applying the rule: Item A's flow ENDS at "Get started" -> Item A is depended on. \
  Item B's flow STARTS at "Get started" -> Item B is the dependent one.
  Step 3 — check whether they share a parent Feature (informative, but not what creates \
  the dependency here).
  Step 4 — check the actual "Linked work items" data given to you: is there an explicit \
  Depends-on/Blocks relation, or only Parent/Child? If only Parent/Child links exist, \
  say so explicitly — a real functional dependency can exist in the requirements even \
  when nothing formally links the two tickets in Azure DevOps.

  Conclusion for the example: "Item B depends on Item A — Item B's flow begins at the \
  exact point Item A's flow ends (the 'Get started' page), so Item A must be delivered \
  first. No formal ADO dependency link exists between them, only parent/child ties to \
  their own Features; worth adding an explicit link."

Apply the same method to any two items you're asked to compare: find the shared point, \
apply the ENDS-there-is-depended-on / STARTS-there-is-dependent rule exactly to \
determine direction (never guess it), check parent Features, then check the given \
"Linked work items" data for an actual Depends-on/Blocks relation. State plainly which \
depends on which — naming the item that STARTS at the shared point as the dependent \
one — and whether that's formally recorded in ADO or only implied by the requirements.
"""

ID_RE = re.compile(r"\b\d{4,}\b")
CORRECT_RE = re.compile(r"^(?:correct|correction)\b[:\s]*(.*)$", re.IGNORECASE)
DEPENDENCY_RE = re.compile(r"\bdepend", re.IGNORECASE)


def load_item(messages: list, user_input: str, pat: str, loaded_ids: set):
    """Returns the loaded item id on success, None on failure."""
    match = ID_RE.search(user_input)
    if not match:
        print(f"Couldn't find a work item number in that. Try: load 97061\n")
        return None
    item_id = match.group()
    print(f"Loading work item #{item_id} from Azure DevOps...")
    work_item, error = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, item_id, pat)
    if work_item is None:
        print(error + "\n")
        # Put the failure in the conversation so a follow-up "why did that fail?"
        # gets a real answer instead of a generic "I have no tool access" reply.
        messages.append({"role": "assistant",
                          "content": f"(Attempted to load work item #{item_id} but it failed: {error})"})
        return None
    loaded_ids.add(item_id)
    related = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, work_item, pat)
    context = build_context_message(work_item, item_id, related)

    item_corrections = build_item_corrections_block(corrections_for_item(load_corrections(), item_id))
    if item_corrections:
        context += "\n\n" + item_corrections

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
    return item_id


def analyse_dependency(messages: list, id_a: str, id_b: str, pat: str, loaded_ids: set):
    """Fetches both items fresh and asks the model to reason about their
    dependency using the worked method in BASE_SYSTEM_PROMPT."""
    print(f"Loading work items #{id_a} and #{id_b} from Azure DevOps...")
    contexts = []
    for item_id in (id_a, id_b):
        work_item, error = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, item_id, pat)
        if work_item is None:
            print(error + "\n")
            return
        loaded_ids.add(item_id)
        related = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, work_item, pat)
        contexts.append(build_context_message(work_item, item_id, related))

    instruction = (
        f"Analyse the dependency between #{id_a} and #{id_b} using the method described "
        f"above. Refer to the two items ONLY as #{id_a} and #{id_b} throughout your answer "
        f"— never substitute a parent Feature's ID or a linked Task's ID for either of "
        f"these two items, even when citing which one a Feature or Task belongs to.\n"
        f"Show your work — do not skip straight to a conclusion:\n"
        f"Step 1: What point (screen, entity, or data) do the two items share, if any?\n"
        f"Step 2: Which item's flow ENDS at that point, and which item's flow STARTS "
        f"there? Apply the rule exactly: ENDS-there = depended on, STARTS-there = "
        f"dependent. Do not skip this step or guess the direction any other way.\n"
        f"Step 3: Do they share a parent Feature? (Note it, but it doesn't decide direction.)\n"
        f"Step 4: Check the 'Linked work items' data above — is there an explicit "
        f"Depends-on/Blocks relation, or only Parent/Child?\n"
        f"Then give your final conclusion in one line: which item depends on which, and "
        f"whether that's formally recorded in ADO or only implied by the requirements."
    )
    messages.append({"role": "user", "content": "\n\n---\n\n".join(contexts) + "\n\n" + instruction})
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
    corrections_context = build_corrections_context(load_corrections())
    if corrections_context:
        system_prompt += "\n\n" + corrections_context

    messages = [{"role": "system", "content": system_prompt}]
    loaded_ids = set()
    current_id = None

    print("task-breakdown chat. Commands: 'load <id>' to load/switch a work item, "
          "'analyse dependency between <id> and <id>', "
          "'correct <note>' to save a correction for future sessions, 'exit' to quit.")
    if len(sys.argv) > 1:
        current_id = load_item(messages, sys.argv[1], pat, loaded_ids) or current_id
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
            # Checked ahead of the load branch: "load 97057 and analyse dependency
            # between 97057 and 97061" mentions "load" too, but dependency intent wins.
            dep_ids = ID_RE.findall(user_input)
            if DEPENDENCY_RE.search(user_input) and len(dep_ids) >= 2:
                analyse_dependency(messages, dep_ids[0], dep_ids[1], pat, loaded_ids)
                continue

            # Treat it as a load request whenever "load" and a work item number
            # appear together anywhere in the message, not just as a prefix —
            # covers phrasing like "now load workitem 97061 and give the title".
            if re.search(r"\bload\b", user_input, re.IGNORECASE) and ID_RE.search(user_input):
                current_id = load_item(messages, user_input, pat, loaded_ids) or current_id
                continue

            correct_match = CORRECT_RE.match(user_input)
            if correct_match:
                note = correct_match.group(1).strip()
                if not note:
                    print("Usage: correct <what was wrong and what it should have been>\n")
                else:
                    save_correction(current_id, note)
                    tag = f"#{current_id}" if current_id else "general"
                    messages.append({"role": "user",
                                      "content": f"(Correction noted for future reference [{tag}]: {note})"})
                    print("Saved — this will also be used in future sessions on this machine.\n")
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
