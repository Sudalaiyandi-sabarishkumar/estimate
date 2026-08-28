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
  export TEAMS_WEBHOOK_URL="..."  # optional -- lets /mom post to a Teams
                                   # channel; see teams_notify.py's docstring

Usage:
  python3 task_breakdown_chat.py
  python3 task_breakdown_chat.py 97061   # load a work item immediately
"""

import json
import re
import subprocess
import sys
import os
from datetime import datetime, timedelta, timezone

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    # Falls back to plain input() -- commands still work, just without the
    # live "/" dropdown, in case someone runs this on a machine where
    # prompt_toolkit isn't installed.
    _PROMPT_TOOLKIT_AVAILABLE = False

from .ado_common import (
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
    strip_html,
)
from .codebase_context import (
    clone_or_update_repo,
    build_codebase_context,
    build_architecture_overview,
    CODEBASE_NUM_CTX,
    CODEBASE_CONTEXT_CHAR_BUDGET,
    ARCHITECTURE_CHAR_BUDGET,
)
from .transcript_utils import load_transcript, chunk_transcript
from .teams_notify import send_to_teams

# Matches the shell alias in ~/.zshrc ("ollama run agent") -- shown in the
# startup banner so it's clear which tool/model you're talking to.
AGENT_NAME = "agent"

REPO_ROLES = ["frontend", "backend", "mobile"]

# BAs on this team tag ticket titles with the platform, e.g. "[Mobile] As a
# User, I want...". Backend is included in both, since a feature almost
# always needs it regardless of which client platform it's on.
PLATFORM_ROLE_MAP = {
    "mobile": (["mobile", "backend"], "Mobile"),
    "ios": (["mobile", "backend"], "Mobile"),
    "android": (["mobile", "backend"], "Mobile"),
    "web": (["frontend", "backend"], "Web"),
    "frontend": (["frontend", "backend"], "Web"),
}


def relevant_roles_for_title(title: str):
    """Returns (roles, platform_label). platform_label is None when the title
    has no recognizable '[Tag]' platform prefix, in which case all three
    roles are requested -- the safest default when the platform can't be
    determined rather than guessing wrong and silently skipping a real repo."""
    m = re.match(r"^\s*\[([^\]]+)\]", title or "")
    if m and m.group(1).strip().lower() in PLATFORM_ROLE_MAP:
        return PLATFORM_ROLE_MAP[m.group(1).strip().lower()]
    return list(REPO_ROLES), None

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

# All commands are slash commands, like Claude Code's own -- '/skills' lists
# them all at runtime (see SKILLS_TEXT below) rather than needing this list
# memorized.
LOAD_RE = re.compile(r"^/load\b", re.IGNORECASE)
CORRECT_RE = re.compile(r"^/(?:correct|correction)\b[:\s]*(.*)$", re.IGNORECASE)
DEPENDENCY_RE = re.compile(r"^/(?:analyse\s+)?dependency\b", re.IGNORECASE)
ESTIMATE_RE = re.compile(r"^/estimate\b", re.IGNORECASE)
SET_REPO_RE = re.compile(r"^/(?:set\s+repo|change\s+repo|new\s+repo)\b", re.IGNORECASE)
SET_ORG_RE = re.compile(r"^/(?:set\s+org|change\s+org|new\s+org)\b", re.IGNORECASE)
MOM_RE = re.compile(r"^/mom\s+(.+)$", re.IGNORECASE)
ADDCALL_RE = re.compile(r"^/(?:addcall|add-call|add\s+call)\b", re.IGNORECASE)
LISTCALLS_RE = re.compile(r"^/(?:listcalls|list-calls|list\s+calls)\b", re.IGNORECASE)
REMOVECALL_RE = re.compile(r"^/(?:removecall|remove-call|remove\s+call|deletecall|delete-call|delete\s+call)\s*(.*)$",
                            re.IGNORECASE)
SKILLS_RE = re.compile(r"^/skills\b", re.IGNORECASE)

# Kept deliberately small, not sized to the context window's actual ceiling.
# A real test against a 15K-char transcript showed the model faithfully
# covering only the first ~third of a single large single-shot input and
# silently thinning out (plus fabricating a due date) on the rest, well
# before num_ctx was actually exhausted -- forcing more, smaller chunks
# keeps each individual call's transcript slice small enough to process
# completely, which matters more here than minimizing the number of calls.
MOM_CHUNK_CHAR_BUDGET = 5000
MOM_NUM_CTX = 12288

SKILLS_TEXT = """Available skills:
  /load <id>                        Load an Azure DevOps work item and get a breakdown
  /estimate <id>                    Codebase-aware estimate (Frontend/Backend/Mobile/Testing)
  /mom <transcript-file>             Minutes of Meeting + action items from a transcript
                                     (optionally posts to Teams -- see TEAMS_WEBHOOK_URL)
  /addcall                          Add a recurring Teams meeting to the automatic
                                     transcript-pull + summarize + post pipeline
  /listcalls                        List configured calls and where each is scheduled
  /removecall <name>                Remove a configured call
  /dependency <id> and <id>         Check dependency direction between two work items
  /correct <note>                   Save a correction, remembered in future sessions
  /set repo                         Change the Frontend/Backend/Mobile repos
  /set org                          Change the Azure DevOps organisation/project
  /skills                           Show this list
  exit (or /exit)                   Quit
Anything else you type is just a normal question to the model."""

# Completion candidates for the live "/" dropdown (see _PROMPT_TOOLKIT_AVAILABLE
# above) -- kept as plain command names/prefixes, not full usage strings, so
# completing one still leaves room to type the actual <id>/<note>/<file> after it.
SKILL_COMMANDS = [
    "/load", "/estimate", "/mom", "/addcall", "/listcalls", "/removecall",
    "/dependency", "/correct", "/set repo", "/set org", "/skills", "/exit",
]


def prompt_org_project(org_state: dict):
    """Session-level org/project selection: confirm the current value if one
    is set (seeded from ado_common's DEFAULT_ORG/DEFAULT_PROJECT), otherwise
    ask for it -- so this tool isn't hardwired to one specific ADO project."""
    if org_state.get("org") and org_state.get("project"):
        print(f"Currently set: {org_state['org']} / {org_state['project']}")
        keep = input("Use this organisation/project? (Y/n): ").strip().lower()
        if keep in ("n", "no"):
            org_state["org"] = None
            org_state["project"] = None
    if not org_state.get("org"):
        org_state["org"] = input("Azure DevOps organisation (e.g. rootquotient): ").strip()
        org_state["project"] = input("Azure DevOps project (e.g. Buddhi Mantra): ").strip()


def _role_resolved(repo_state: dict, role: str) -> bool:
    """A role counts as settled for the session once it has a URL, or was
    explicitly skipped -- either way it shouldn't be re-asked every time."""
    entry = repo_state.get(role, {})
    return bool(entry.get("url")) or entry.get("skipped", False)


def prompt_repos(repo_state: dict, roles: list):
    """Session-level repo/branch selection for the given roles only --
    confirm the current set if already resolved, otherwise ask for each.
    Leaving a URL blank (or typing 'skip') skips that role entirely -- not
    every team has all three repos. `roles` lets estimate_with_codebase pass
    just the ones relevant to a given ticket's platform (e.g. skip asking
    about Frontend for a ticket titled '[Mobile] ...')."""
    if all(_role_resolved(repo_state, role) for role in roles):
        print("Currently set:")
        for role in roles:
            r = repo_state[role]
            shown = "(skipped)" if r.get("skipped") else f"{r['url']} @ {r['branch']}"
            print(f"  {role.capitalize()}: {shown}")
        keep = input("Use these repos/branches for this estimate? (Y/n): ").strip().lower()
        if keep in ("n", "no"):
            for role in roles:
                repo_state[role] = {"url": None, "branch": None, "skipped": False}
    for role in roles:
        if not _role_resolved(repo_state, role):
            url = input(f"{role.capitalize()} repo URL "
                        f"(e.g. https://dev.azure.com/org/project/_git/repo, "
                        f"or leave blank to skip): ").strip()
            if not url or url.lower() == "skip":
                repo_state[role] = {"url": None, "branch": None, "skipped": True}
                continue
            branch = input(f"{role.capitalize()} branch (e.g. release/stable): ").strip()
            repo_state[role] = {"url": url, "branch": branch, "skipped": False}

# Split into two calls on purpose: the implementation-status question is a
# categorical decision (is it built or not?) that needs to be answered the
# same way every time, so it runs at temperature=0 (deterministic) in its own
# call. The estimate/analysis that may follow is free-text and benefits from
# the normal temperature=0.3 for natural-sounding output. Mixing the two in
# one sampled call let the model's answer to the yes/no question drift
# between runs even when the underlying grepped evidence never changed.
def build_status_check_instruction(relevant_roles: list) -> str:
    section_names = ", ".join(f"'=== {r.upper()} REPO ==='" for r in relevant_roles)
    return (
    "Based ONLY on the CODEBASE CONTEXT above (not the ticket's own state/comments), "
    "determine whether this SPECIFIC user story's functionality is already built. "
    f"CODEBASE CONTEXT is divided into {section_names} section(s) — only the repos "
    "relevant to this ticket's platform are included, and a feature can be fully "
    "implemented in just one of them even if another shows nothing relevant. "
    "A section marked '(skipped — no repo provided for this role, not checked)' means "
    "that layer was never looked at, which is NOT the same as evidence it doesn't exist "
    "— base your verdict only on the sections that were actually checked, and don't "
    "treat a skipped section as proof anything is missing.\n\n"
    "Being thematically related is NOT enough. A file that only displays static text, "
    "labels, or images about the same topic (e.g. an instructions/introduction screen "
    "that just names or lists the feature) does NOT count as implementing it. Real "
    "matching code must perform the actual behavior the acceptance criteria describes "
    "(the specific calculation, interactive flow, form fields, or business logic) — not "
    "just be about the same subject.\n\n"
    "Step 1: Name ONE concrete, checkable requirement from the acceptance criteria (a "
    "specific calculation, interaction, or piece of logic — not just a screen/topic name).\n"
    "Step 2: Check whether the CODEBASE CONTEXT's actual code snippets (not just file or "
    "class names) perform that requirement.\n"
    "Step 3: Decide based on Step 2 —\n"
    "- The snippets show the real required behavior fully -> Already Implemented.\n"
    "- The snippets show some of it but not all (e.g. only a related instructional/intro "
    "screen exists, but not the interactive/calculated logic itself) -> Partially Implemented.\n"
    "- CODEBASE CONTEXT says no files were found, OR the files it lists are only "
    "thematically related without performing the actual required behavior -> Not Implemented.\n\n"
    "You are NOT allowed to write \"no matching code found\" as your EVIDENCE if the "
    "CODEBASE CONTEXT above actually lists file paths — if you conclude Not Implemented "
    "despite files being listed, your EVIDENCE must name those files and explain "
    "specifically why they don't perform the required behavior (e.g. \"only an "
    "introductory screen listing pillar names exists, not the interactive scored "
    "assessment logic\"), not simply claim nothing was found.\n\n"
    "Respond with ONLY these two lines, nothing else, no other commentary:\n"
    "IMPLEMENTATION STATUS: <Already Implemented | Partially Implemented | Not Implemented>\n"
    "EVIDENCE: <one sentence naming the specific requirement you checked and what the code "
    "actually does or doesn't do about it>"
)

STATUS_LINE_RE = re.compile(
    r"IMPLEMENTATION STATUS\**:?\**\s*(Already Implemented|Partially Implemented|Not Implemented)",
    re.IGNORECASE,
)
EVIDENCE_LINE_RE = re.compile(r"EVIDENCE\**:?\**\s*(.+)", re.IGNORECASE)


def parse_status(reply: str):
    """Returns (status, evidence). status is None if the reply didn't follow
    the required format at all -- callers should treat that as 'unknown' and
    default to the safe path (full estimate) rather than silently skipping it."""
    m = STATUS_LINE_RE.search(reply or "")
    status = m.group(1) if m else None
    e = EVIDENCE_LINE_RE.search(reply or "")
    evidence = e.group(1).strip() if e else ""
    return status, evidence


def build_estimate_instruction(item_id: str, status: str, evidence: str,
                                scope_instruction: str, relevant_roles: list) -> str:
    """Categories offered for the hour breakdown are built strictly from
    relevant_roles -- e.g. 'Frontend' is never even mentioned for a
    Mobile-only ticket. Telling the model "Frontend doesn't apply, don't use
    it" while still listing it as an available category didn't work reliably
    (it kept giving Frontend hours anyway); not naming it as an option at all
    is the structural fix."""
    section_names = ", ".join(f"'=== {r.upper()} REPO ==='" for r in relevant_roles)
    categories = [r.capitalize() for r in relevant_roles] + ["Testing"]
    category_list = ", ".join(categories[:-1]) + f", and {categories[-1]}"

    if "mobile" in relevant_roles and "frontend" in relevant_roles:
        platform_note = (
            " IMPORTANT — 'Frontend' here means ONLY the separate web repo, NOT "
            "'client-side UI' in general; any screen/icon/button/UI logic for the "
            "mobile app belongs under Mobile, never under Frontend, even though 'UI "
            "work' might normally make you think 'Frontend'."
        )
    elif "mobile" in relevant_roles:
        platform_note = (
            " This ticket only involves Mobile and Backend — there is no Frontend "
            "(web) repo in scope, so ALL UI/screen/icon work goes under Mobile; do "
            "not create a separate Frontend category or heading at all."
        )
    elif "frontend" in relevant_roles:
        platform_note = (
            " This ticket only involves Frontend (web) and Backend — there is no "
            "Mobile repo in scope, so do not create a separate Mobile category or "
            "heading at all."
        )
    else:
        platform_note = ""

    return (
        f"The implementation status for #{item_id} has already been determined:\n"
        f"IMPLEMENTATION STATUS: {status}\n"
        f"EVIDENCE: {evidence}\n\n"
        f"Using the CODEBASE CONTEXT above together with the WORK ITEM CONTEXT, produce a "
        f"codebase-aware estimate for #{item_id}. {scope_instruction}\n\n"
        f"CODEBASE CONTEXT and ARCHITECTURE OVERVIEW are each divided into {section_names} "
        f"section(s), from the real repo(s) relevant to this ticket's platform.{platform_note} "
        f"A repo showing nothing relevant just means this feature doesn't touch that part "
        f"of the stack, not that the analysis is incomplete.\n\n"
        f"Before deciding there's no Backend work: check the BACKEND REPO section of "
        f"ARCHITECTURE OVERVIEW above.\n"
        f"- If it's marked '(skipped — no repo provided for this role, not checked)': you "
        f"have NO evidence either way. Do NOT conclude 'no backend work needed' and do NOT "
        f"invent a justification for skipping it (e.g. 'this only updates local state') "
        f"unless the ticket itself is unambiguous that the feature is purely local/device-"
        f"only with no login, account, or cross-session persistence involved anywhere in "
        f"its wording. If the ticket gates the feature behind being logged in/registered, "
        f"or implies the data should survive across sessions/devices/reinstalls (phrases "
        f"like 'save for later', 'sync', 'my account'), treat that as a strong signal "
        f"backend work IS needed even with zero code evidence, and write 'Backend: likely "
        f"needed (backend repo not provided to confirm — assumed from account/persistence "
        f"requirements in the ticket)' with a real hour estimate, not 'N/A' or 'not required'.\n"
        f"- If it was actually checked: CODEBASE CONTEXT is grepped from the ticket's own "
        f"wording, so for a brand-new feature it will naturally show little or nothing about "
        f"backend needs -- that absence does NOT mean no backend work is needed either. If "
        f"the backend repo already has an API/service/repository layer, and this feature "
        f"involves data that should persist per-user or across sessions/devices, assume it "
        f"needs a backend endpoint following that same established pattern, even with no "
        f"existing code for this specific feature yet.\n\n"
        f"Give hour estimates (S = under 4h, M = 4-16h, L = over 16h) for one mid-level "
        f"engineer, grouped ONLY by {category_list} — do not add any other category, even "
        f"one you might normally expect for a software estimate. If a repo for one of "
        f"these categories was skipped/not provided, still include that category's heading "
        f"and say estimation wasn't possible for it rather than omitting it silently.\n\n"
        f"Then give a CODEBASE IMPACT ANALYSIS, as its own labeled section, covering exactly "
        f"these six points:\n"
        f"1. Which existing modules/components/services will be affected.\n"
        f"2. Whether this could impact existing functionality.\n"
        f"3. Potential regression areas.\n"
        f"4. Dependencies or areas that may require changes.\n"
        f"5. Technical risks/challenges identified from the current implementation.\n"
        f"6. Additional work not obvious from the user story alone.\n\n"
        f"Only reference files/modules that actually appear in the CODEBASE CONTEXT above — "
        f"if it didn't surface anything relevant to a point, say so rather than inventing "
        f"file or module names."
    )


def load_item(messages: list, user_input: str, pat: str, loaded_ids: set, org_state: dict):
    """Returns the loaded item id on success, None on failure."""
    match = ID_RE.search(user_input)
    if not match:
        print(f"Couldn't find a work item number in that. Try: /load 97061\n")
        return None
    item_id = match.group()
    print(f"Loading work item #{item_id} from Azure DevOps...")
    work_item, error = fetch_work_item(org_state["org"], org_state["project"], item_id, pat)
    if work_item is None:
        print(error + "\n")
        # Put the failure in the conversation so a follow-up "why did that fail?"
        # gets a real answer instead of a generic "I have no tool access" reply.
        messages.append({"role": "assistant",
                          "content": f"(Attempted to load work item #{item_id} but it failed: {error})"})
        return None
    loaded_ids.add(item_id)
    related = fetch_related_context(org_state["org"], org_state["project"], work_item, pat)
    context = build_context_message(work_item, item_id, related)

    item_corrections = build_item_corrections_block(corrections_for_item(load_corrections(), item_id))
    if item_corrections:
        context += "\n\n" + item_corrections

    # Whatever the user asked for beyond "/load <id>" becomes the actual request,
    # e.g. "/load 97061 and give only the title" -> "and give only the title".
    instruction = re.sub(r"^/load\b", "", user_input, flags=re.IGNORECASE)
    instruction = re.sub(r"\bwork\s*items?\b", "", instruction, flags=re.IGNORECASE)
    instruction = instruction.replace(item_id, "")
    instruction = re.sub(r"\s+", " ", instruction).strip(" ,./")
    if not instruction:
        instruction = "Give me a full breakdown."

    messages.append({"role": "user", "content": context + "\n\n" + instruction})
    print()
    reply = call_ollama(messages, DEFAULT_MODEL)
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    print()
    return item_id


def analyse_dependency(messages: list, id_a: str, id_b: str, pat: str, loaded_ids: set, org_state: dict):
    """Fetches both items fresh and asks the model to reason about their
    dependency using the worked method in BASE_SYSTEM_PROMPT."""
    print(f"Loading work items #{id_a} and #{id_b} from Azure DevOps...")
    contexts = []
    for item_id in (id_a, id_b):
        work_item, error = fetch_work_item(org_state["org"], org_state["project"], item_id, pat)
        if work_item is None:
            print(error + "\n")
            return
        loaded_ids.add(item_id)
        related = fetch_related_context(org_state["org"], org_state["project"], work_item, pat)
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


# Phase 1 of the Governance & Scrum MoM + Action Items agent: reads a
# manually-provided transcript file (e.g. exported from Teams as .vtt, or
# plain .txt). Phase 2 -- once a Teams/M365 tenant admin enables Graph
# transcript API access and grants an app OnlineMeetingTranscript.Read.All,
# which is an organizational prerequisite outside this tool's control, not
# something this code can do on its own -- will fetch the transcript
# automatically instead of reading a file, but generate_mom() itself won't
# need to change since it only cares about the resulting plain text.
MOM_SYSTEM_PROMPT = """You are a professional meeting scribe producing Minutes of Meeting \
(MoM) for a governance or scrum call, from a real transcript.

Base everything ONLY on the transcript content you are given — never invent attendees, \
decisions, or action items that weren't actually said.

Owner labeling rule — the word "Unidentified" must NEVER appear on the same line as an \
actual name, in any form, including in parentheses. Pick exactly one of these two forms, \
nothing else:
- The transcript names who said it (e.g. a line starting "Sudalaiyandi: I'll..." or \
"Arjun: I'll...") -> write ONLY the bare name: "Owner: Sudalaiyandi". Correct examples: \
"Owner: Arjun", "Owner: Priya". WRONG, never do this: "Owner: Unidentified speaker \
(Sudalaiyandi)" -- if you know the name, "Unidentified" does not belong anywhere near it.
- The transcript never names who said it at all -> write ONLY "Owner: Unidentified \
speaker", with no name anywhere on that line.

Action item rule: only count something as an ACTION ITEM if it's a real, concrete \
commitment someone made or was assigned to DO something by a specific point in time or \
before a next step — phrases like "I'll do X by Thursday" or "I'll send Y tomorrow" \
qualify. A plain FYI/heads-up with no commitment attached (e.g. "just so you know, we're \
planning to upgrade the SDK next sprint") does NOT qualify as an action item, even if it's \
worth noting — put that under Key Discussion Points instead, not Action Items.
"""

MOM_INSTRUCTION = """Produce the Minutes of Meeting in exactly this structure:

## Meeting Summary
A concise paragraph (3-6 sentences) covering what was discussed and any decisions made.

## Key Discussion Points
Bullet list of the main topics/updates raised, grouped logically. FYI/heads-up items with \
no owner commitment belong here, not in Action Items.

## Action Items
One line per item, in exactly this format (use "not specified" if a field isn't in the \
transcript):
- [ ] <action description> — Owner: <name> — Due: <date>

## Open Questions / Blockers
Anything raised as unresolved or blocking. Write "None raised" if there weren't any.
"""

MOM_CHUNK_INSTRUCTION = """This is one part of a longer meeting transcript (not the whole \
meeting). Respond in EXACTLY this format, with these two headers verbatim (a later step \
mechanically merges these sections across parts by searching for these exact header \
strings, so do not rename, reformat, or omit either one even if a section is empty):

## DISCUSSION POINTS
- <topic raised here, one bullet per topic>

## ACTION ITEMS
- [ ] <action description> — Owner: <name> — Due: <date, or "not specified">

Do not write a full MoM yet — this is raw material for a later consolidation pass, so list \
everything covered in THIS part, however minor; nothing gets a second chance to be included.

If this part contains no real meeting dialogue at all (e.g. it's empty, only a file header, \
a trailing note, or unrelated text with no one actually speaking), still use both headers, \
each followed by a single line: "None." Never invent discussion points or action items to \
fill the space.
"""

# No LLM-based reduce anymore. Repeated testing showed that asking the model
# to merge 4 parts' worth of already-correct notes into one final MoM reliably
# dropped and occasionally misattributed real content, even at temperature=0
# with an explicit "never drop anything" instruction -- a genuine capability
# ceiling for this model size at multi-source synthesis, not a prompt-wording
# problem (see conversation). merge_discussion_and_action_items() below
# concatenates the DISCUSSION POINTS/ACTION ITEMS sections mechanically in
# plain Python instead, which cannot drop or reword anything since there's no
# model involved in that step. The one remaining LLM call
# (MOM_SUMMARY_INSTRUCTION) only writes the intro paragraph and checks for
# open questions, working from the already-complete merged list -- a
# summarization task, not a selection task, so nothing it does can cause the
# real content to go missing.
MOM_SUMMARY_INSTRUCTION = """Below is the COMPLETE, already-finalized list of discussion \
points and action items for this meeting (assembled separately, not by you). Do not add, \
remove, reword, or reorder any of it.

Using only this list, write:

## Meeting Summary
A concise paragraph (3-6 sentences) covering what was discussed and any decisions made.

## Open Questions / Blockers
Anything in the list above that reads as unresolved or blocking. Write "None raised" if \
there weren't any.

Output ONLY those two sections, nothing else -- the discussion points and action items are \
appended after your response, not written by you.
"""


def merge_discussion_and_action_items(part_notes: list) -> tuple:
    """Mechanically extracts the '## DISCUSSION POINTS' / '## ACTION ITEMS'
    bullets from each part's chunk-instruction output (see MOM_CHUNK_INSTRUCTION
    for the exact header contract) and concatenates them -- no LLM involved,
    so nothing can be silently dropped or misattributed the way the old
    LLM-based reduce step repeatedly was. Returns (discussion_points_md,
    action_items_md), each already-formatted markdown bullet lists."""
    discussion_bullets, action_bullets = [], []
    seen_discussion, seen_action = set(), set()

    for note in part_notes:
        for header, bullets, seen in (
            ("## DISCUSSION POINTS", discussion_bullets, seen_discussion),
            ("## ACTION ITEMS", action_bullets, seen_action),
        ):
            start = note.find(header)
            if start == -1:
                continue
            start += len(header)
            end = note.find("##", start)
            section = note[start:end if end != -1 else None]
            for line in section.splitlines():
                line = line.strip()
                if not line or line.lower() in ("none.", "none", "- none.", "- none"):
                    continue
                if not line.startswith("-"):
                    continue
                # Exact-duplicate-only dedup (case/whitespace-insensitive) --
                # deliberately not "smart": two items only merge if they're
                # the same string, never based on the model judging them
                # similar, which is exactly the judgment call that kept going
                # wrong in the old reduce step.
                key = " ".join(line.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                bullets.append(line)

    discussion_md = "\n".join(discussion_bullets) if discussion_bullets else "- None raised."
    action_md = "\n".join(action_bullets) if action_bullets else "- None."
    return discussion_md, action_md


def run_mom_pipeline(text: str, base_name: str, messages: list = None):
    """Core MoM generation: chunk -> per-chunk extraction -> mechanical merge
    -> summary -> save to mom_output/<base_name>_mom.md. Returns (out_path,
    reply). Shared by the interactive /mom <file> command (passes `messages`
    so the result joins the chat history) and auto_mom.py's unattended
    Teams-transcript pipeline (no chat session, so `messages` stays None) --
    posting to Teams is the caller's decision, not made here, since the two
    callers want different UX for it (ask vs. always)."""
    chunks = chunk_transcript(text, max_chars=MOM_CHUNK_CHAR_BUDGET)
    mom_messages = [{"role": "system", "content": MOM_SYSTEM_PROMPT}]

    if len(chunks) == 1:
        print("Generating Minutes of Meeting...")
        prompt = f"TRANSCRIPT:\n{chunks[0]}\n\n{MOM_INSTRUCTION}"
        reply = call_ollama(mom_messages + [{"role": "user", "content": prompt}],
                             DEFAULT_MODEL, num_ctx=MOM_NUM_CTX)
        if reply is None:
            print()
            return None, None
    else:
        print(f"Transcript is long ({len(chunks)} parts) — summarizing each part first...")
        part_notes = []
        for i, chunk in enumerate(chunks):
            print(f"  Part {i + 1}/{len(chunks)}...")
            prompt = f"TRANSCRIPT PART {i + 1} of {len(chunks)}:\n{chunk}\n\n{MOM_CHUNK_INSTRUCTION}"
            # temperature=0: this is faithful extraction ("what was said in this
            # chunk"), not creative writing -- sampling variance here was shown
            # to change which items get extracted and even who gets named as
            # owner on the exact same transcript across runs, so it gets the
            # same determinism treatment as the summary call below.
            part_reply = call_ollama(mom_messages + [{"role": "user", "content": prompt}],
                                      DEFAULT_MODEL, num_ctx=MOM_NUM_CTX, show_progress=False,
                                      temperature=0)
            part_notes.append(part_reply or "")

        # Mechanical merge (plain Python, no model involved) -- see
        # merge_discussion_and_action_items()'s docstring for why: the
        # LLM-based reduce this replaced reliably dropped and occasionally
        # misattributed real content across repeated testing, even at
        # temperature=0 with an explicit "never drop anything" instruction.
        discussion_md, action_md = merge_discussion_and_action_items(part_notes)

        print("Writing summary from the complete, merged list...")
        merged_list_text = f"## DISCUSSION POINTS\n{discussion_md}\n\n## ACTION ITEMS\n{action_md}"
        prompt = f"{merged_list_text}\n\n{MOM_SUMMARY_INSTRUCTION}"
        summary_reply = call_ollama(mom_messages + [{"role": "user", "content": prompt}],
                                     DEFAULT_MODEL, num_ctx=MOM_NUM_CTX, temperature=0)
        if summary_reply is None:
            print()
            return None, None

        # MOM_SUMMARY_INSTRUCTION asks for "## Meeting Summary" followed by
        # "## Open Questions / Blockers", in that order -- but the canonical
        # MoM order (MOM_INSTRUCTION, the single-chunk path) is Summary ->
        # Discussion Points -> Action Items -> Open Questions. Splice the
        # mechanical sections in between rather than appending them after
        # both, or Open Questions ends up sitting right after the summary
        # paragraph, ahead of the actual discussion/action content.
        summary_reply = summary_reply.strip()
        # Match on the heading regardless of how many #'s the model used --
        # it doesn't reliably match MOM_SUMMARY_INSTRUCTION's own "##" depth
        # (often writes "### Open Questions" instead, echoing its own
        # "### Meeting Summary" heading a few lines earlier). A plain
        # substring search for "## Open Questions" still matches inside that
        # "###" one character in, leaving a stray lone "#" dangling at the
        # end of the summary paragraph.
        heading_match = re.search(r"^#+\s*Open Questions", summary_reply, re.MULTILINE)
        if heading_match:
            idx = heading_match.start()
            meeting_summary_part = summary_reply[:idx].strip()
            open_questions_part = summary_reply[idx:].strip()
        else:
            meeting_summary_part = summary_reply
            open_questions_part = "## Open Questions / Blockers\nNone raised."

        tail = (
            f"\n\n## Key Discussion Points\n{discussion_md}\n\n"
            f"## Action Items\n{action_md}\n\n"
            f"{open_questions_part}\n"
        )
        print(tail)
        reply = meeting_summary_part + tail

    if messages is not None:
        messages.append({"role": "user", "content": f"(Generated MoM from transcript: {base_name})"})
        messages.append({"role": "assistant", "content": reply})

    os.makedirs("mom_output", exist_ok=True)
    out_path = os.path.join("mom_output", f"{base_name}_mom.md")
    with open(out_path, "w") as f:
        f.write(reply)
    print(f"\nSaved to {out_path}\n")

    return out_path, reply


def generate_mom(messages: list, user_input: str):
    """/mom <path-to-transcript-file>: the interactive, manually-triggered
    path -- reads a local transcript file, runs it through run_mom_pipeline(),
    and (unlike auto_mom.py's unattended pipeline) asks before posting
    anywhere, since a live chat session has a human right there to answer."""
    match = MOM_RE.match(user_input)
    if not match:
        print("Usage: /mom <path-to-transcript-file>\n")
        return
    path = match.group(1).strip()

    text, error = load_transcript(path)
    if text is None:
        print(error + "\n")
        return

    base = os.path.splitext(os.path.basename(path))[0]
    out_path, reply = run_mom_pipeline(text, base, messages)
    if reply is None:
        return

    offer_teams_post(f"Minutes of Meeting — {base}", reply)


def estimate_with_codebase(messages: list, user_input: str, pat: str,
                            loaded_ids: set, repo_state: dict, org_state: dict):
    """Like load_item, but also clones the Frontend/Backend/Mobile repos (once
    per session, cached after that) and greps each for terms from the ticket,
    so the estimate accounts for the actual codebase, not just the ticket text."""
    match = ID_RE.search(user_input)
    if not match:
        print("Couldn't find a work item number in that. Try: /estimate 97061\n")
        return
    item_id = match.group()

    # Fetch the ticket first (before asking about repos) so its title can
    # decide which repos are even relevant -- a BA-tagged "[Mobile] ..." or
    # "[Web] ..." title means we only need to ask about that platform's repo
    # plus Backend, not all three every time.
    print(f"Loading work item #{item_id} from Azure DevOps...")
    work_item, error = fetch_work_item(org_state["org"], org_state["project"], item_id, pat)
    if work_item is None:
        print(error + "\n")
        return
    loaded_ids.add(item_id)
    related = fetch_related_context(org_state["org"], org_state["project"], work_item, pat)
    ticket_context = build_context_message(work_item, item_id, related)

    fields = work_item.get("fields", {})
    title = fields.get("System.Title", "")
    search_text = title + " " + strip_html(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", ""))

    relevant_roles, platform_label = relevant_roles_for_title(title)
    if platform_label:
        print(f"Ticket title indicates a {platform_label} task — will ask about "
              f"{' and '.join(r.capitalize() for r in relevant_roles)} only.\n")
    else:
        print("Couldn't tell the platform from the ticket title — asking about all three repos.\n")

    prompt_repos(repo_state, relevant_roles)

    if all(repo_state[role].get("skipped") for role in relevant_roles):
        print("No repos provided — falling back to a plain ticket-based breakdown "
              "(same as /load), since there's nothing to check the codebase against.\n")
        load_item(messages, f"/load {item_id}", pat, loaded_ids, org_state)
        return

    # Each relevant repo gets a share of the overall context budget so
    # multiple repos' worth of matches don't blow past CODEBASE_NUM_CTX the
    # way one repo's full budget x N would.
    per_repo_char_budget = CODEBASE_CONTEXT_CHAR_BUDGET // len(relevant_roles)
    codebase_blocks = []
    repo_paths = {}
    # Irrelevant roles (e.g. Frontend on a Mobile-only ticket) are left out of
    # the prompt entirely, not just marked "(not applicable)" -- the model kept
    # giving Frontend hours anyway when it was merely told not to, so it's
    # removed structurally instead of relying on that instruction being followed.
    for role in relevant_roles:
        r = repo_state[role]
        if r.get("skipped"):
            # Not the same as "grepped and found nothing" -- this layer was
            # never checked at all, which matters for the status/backend
            # reasoning below (absence of evidence isn't evidence of absence).
            codebase_blocks.append(f"=== {role.upper()} REPO ===\n(skipped — no repo "
                                    f"provided for this role, not checked)")
            continue
        print(f"Cloning/updating {role} repo: {r['url']} @ {r['branch']}...")
        repo_path, error = clone_or_update_repo(r["url"], r["branch"], pat)
        if repo_path is None:
            print(f"Couldn't get the {role} repo: {error}\n")
            # Clear so a typo'd URL/branch isn't silently reused on the next attempt.
            repo_state[role] = {"url": None, "branch": None, "skipped": False}
            return
        repo_paths[role] = repo_path
        print(f"Searching {role} repo for relevant files...")
        block = build_codebase_context(repo_path, search_text, char_budget=per_repo_char_budget, title=title)
        codebase_blocks.append(f"=== {role.upper()} REPO ===\n{block}")
    codebase_block = "\n\n".join(codebase_blocks)

    # Call 1: deterministic (temperature=0) yes/no-style classification, kept
    # separate from the free-text estimate below -- this categorical decision
    # needs to come out the same way every time given the same evidence, which
    # sampling at the normal temperature does not reliably do (see conversation).
    status_prompt = f"{ticket_context}\n\n{codebase_block}\n\n{build_status_check_instruction(relevant_roles)}"
    status_messages = [messages[0], {"role": "user", "content": status_prompt}]
    print("Checking implementation status...")
    status_reply = call_ollama(status_messages, DEFAULT_MODEL, num_ctx=CODEBASE_NUM_CTX,
                                temperature=0, show_progress=False)
    status, evidence = parse_status(status_reply)
    if status is None:
        status, evidence = "Not Implemented", "couldn't parse a clear status from the model -- defaulting to a full estimate rather than silently skipping it"

    print(f"IMPLEMENTATION STATUS: {status}")
    print(f"EVIDENCE: {evidence}\n")

    if status.lower() == "already implemented":
        messages.append({"role": "user", "content": status_prompt})
        messages.append({"role": "assistant", "content": status_reply})
        return

    # Call 2: only reached for Not Implemented / Partially Implemented. Normal
    # sampling temperature is fine here -- it's free-text estimate wording,
    # not a categorical decision.
    scope_instruction = (
        "Estimate the FULL feature from scratch."
        if status.lower() == "not implemented"
        else "Estimate ONLY the missing/remaining work, not the whole feature -- "
             "say explicitly what's already done vs. what's left."
    )
    # A new/missing feature's ticket vocabulary can't match its own
    # not-yet-written backend code, so build_codebase_context alone is blind
    # to whether backend work is needed -- add each repo's general API/service
    # layer as a separate signal so the model can reason from established
    # patterns (e.g. "the backend repo already exposes endpoints elsewhere")
    # instead of just the absence of feature-specific matches.
    per_repo_arch_budget = ARCHITECTURE_CHAR_BUDGET // len(relevant_roles)
    architecture_blocks = []
    for role in relevant_roles:
        if role not in repo_paths:
            architecture_blocks.append(f"=== {role.upper()} REPO ===\n(skipped — no repo "
                                        f"provided for this role, not checked)")
            continue
        arch = build_architecture_overview(repo_paths[role], char_budget=per_repo_arch_budget)
        architecture_blocks.append(f"=== {role.upper()} REPO ===\n{arch}")
    architecture_block = "\n\n".join(architecture_blocks)
    instruction = build_estimate_instruction(
        item_id=item_id, status=status, evidence=evidence,
        scope_instruction=scope_instruction, relevant_roles=relevant_roles)
    user_turn = f"{ticket_context}\n\n{codebase_block}\n\n{architecture_block}\n\n{instruction}"

    # Bounded call: system prompt + this turn only, not the full accumulated
    # session history, so this heaviest command's context usage doesn't grow
    # with how many tickets were loaded earlier in the session.
    call_messages = [messages[0], {"role": "user", "content": user_turn}]
    print()
    reply = call_ollama(call_messages, DEFAULT_MODEL, num_ctx=CODEBASE_NUM_CTX)
    if reply is not None:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": reply})
    print()


def require_pat(pat: str) -> bool:
    """ADO-touching commands (load/estimate/analyse dependency) need a PAT;
    mom/correct/general questions don't, so this is checked per-command
    rather than at startup -- someone who only wants 'mom' shouldn't need
    Azure DevOps access set up at all."""
    if pat:
        return True
    print("AZURE_DEVOPS_PAT is not set. Run: export AZURE_DEVOPS_PAT=\"...\"\n")
    return False


def offer_teams_post(title: str, reply: str):
    """After a MoM is generated, ask whether to post it to the Teams channel
    via TEAMS_WEBHOOK_URL (see teams_notify.py's docstring for one-time
    setup). Skipped silently on 'n' -- posting to a shared channel is the
    one action this whole tool takes outside the local machine, so it's
    opt-in per run, never automatic."""
    post = input("Post this MoM to the Teams channel? (y/N): ").strip().lower()
    if post not in ("y", "yes"):
        return
    webhook_url = os.environ.get("TEAMS_WEBHOOK_URL")
    if not webhook_url:
        print("TEAMS_WEBHOOK_URL is not set -- see teams_notify.py's docstring "
              "for one-time setup (add a Workflows webhook to the channel, "
              "then export TEAMS_WEBHOOK_URL=\"...\"). Skipping.\n")
        return
    ok, error = send_to_teams(webhook_url, title, reply)
    if ok:
        print("Posted to Teams.\n")
    else:
        print(f"Couldn't post to Teams: {error}\n")


# Kept as a literal here rather than imported from auto_mom.py, which
# imports run_mom_pipeline from *this* module -- importing back the other
# way would be circular. Keep this in sync with auto_mom.MEETINGS_CONFIG_FILE
# if that ever changes.
MEETINGS_CONFIG_FILE = "meetings_config.json"

_DAY_ALIASES = {
    "sun": 0, "sunday": 0, "mon": 1, "monday": 1, "tue": 2, "tues": 2,
    "tuesday": 2, "wed": 3, "wednesday": 3, "thu": 4, "thurs": 4,
    "thursday": 4, "fri": 5, "friday": 5, "sat": 6, "saturday": 6,
}


def _parse_days_to_cron(text: str):
    """Returns (cron_dow_field, error). Accepts shortcuts ('daily',
    'weekdays', 'weekends') or a comma/space-separated list of day
    names/abbreviations, and turns either into a cron day-of-week field
    (0=Sunday..6=Saturday)."""
    text = text.strip().lower()
    if text in ("daily", "everyday", "every day"):
        return "*", None
    if text in ("weekdays", "mon-fri", "monday-friday"):
        return "1-5", None
    if text in ("weekends", "sat-sun", "saturday-sunday"):
        return "0,6", None
    days = []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if part not in _DAY_ALIASES:
            return None, (f"Didn't recognize day '{part}'. Use day names (mon, tuesday, ...), "
                           f"'weekdays', 'weekends', or 'daily'.")
        days.append(_DAY_ALIASES[part])
    if not days:
        return None, "No days given."
    return ",".join(str(d) for d in sorted(set(days))), None


def _parse_time(text: str):
    """Returns (hour, minute, error) on a 24h clock, accepting '11:00',
    '11:00 AM', '3:30pm', or '15:30'."""
    text = text.strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%H"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.hour, dt.minute, None
        except ValueError:
            continue
    return None, None, f"Didn't recognize time '{text}'. Try '11:00' (24h) or '11:00 AM'."


def _expand_cron_dow(cron_dow: str) -> set:
    """Reverses _parse_days_to_cron's output back into a set of weekday
    ints (0=Sunday..6=Saturday) -- only needs to handle the shapes that
    function actually produces: '*', 'a-b', or a comma-separated list."""
    if cron_dow == "*":
        return set(range(7))
    days = set()
    for part in cron_dow.split(","):
        if "-" in part:
            start, end = part.split("-")
            days.update(range(int(start), int(end) + 1))
        else:
            days.add(int(part))
    return days


def _local_to_utc_cron(cron_dow_local: str, hour: int, minute: int):
    """Converts a local day-of-week set + local time into the UTC
    equivalents GitHub Actions' `schedule:` cron needs (it's always UTC,
    regardless of where the workflow runs). Handles the day rolling over
    (e.g. a late-night local time landing on the *previous* UTC day) by
    converting one reference date through the system's actual local
    timezone rather than assuming a fixed offset -- correct even if this
    ever runs somewhere with DST, though IST (this machine) doesn't have it.
    Returns (utc_cron_dow, utc_hour, utc_minute)."""
    local_tz = datetime.now().astimezone().tzinfo
    reference = datetime(2024, 1, 1, hour, minute, tzinfo=local_tz)  # a Monday
    reference_utc = reference.astimezone(timezone.utc)
    day_shift = (reference_utc.date() - reference.date()).days

    local_days = _expand_cron_dow(cron_dow_local)
    utc_days = {(d + day_shift) % 7 for d in local_days}
    utc_cron_dow = ",".join(str(d) for d in sorted(utc_days))
    return utc_cron_dow, reference_utc.hour, reference_utc.minute


def add_scheduled_call(user_input: str):
    """/addcall: interactively collects everything auto_mom.py needs for one
    recurring meeting's automatic pull-transcript + summarize + post
    pipeline -- join URL, organizer, target channel webhook, and when to
    check for it -- then writes it to meetings_config.json and offers to
    install the matching crontab line. Nothing is written or installed
    until you confirm; cancel at any prompt by leaving it blank."""
    print("Adding a new recurring call to the automatic MoM pipeline.\n")

    name = input("Short name for this meeting (e.g. daily-standup, no spaces): ").strip()
    if not name:
        print("Cancelled -- no name given.\n")
        return
    name = re.sub(r"\s+", "-", name)

    meetings = []
    if os.path.exists(MEETINGS_CONFIG_FILE):
        with open(MEETINGS_CONFIG_FILE) as f:
            meetings = json.load(f)
    if any(m.get("name") == name for m in meetings):
        overwrite = input(f"'{name}' already exists in {MEETINGS_CONFIG_FILE} "
                           f"-- overwrite it? (y/N): ").strip().lower()
        if overwrite not in ("y", "yes"):
            print("Cancelled.\n")
            return
        meetings = [m for m in meetings if m.get("name") != name]

    print("\nOrganizer's Entra object ID -- whoever organizes this recurring meeting.")
    print("Find it: Entra admin center -> Users -> that person -> the 'Object ID' field.")
    organizer_user_id = input("Organizer's object ID: ").strip()
    if not organizer_user_id:
        print("Cancelled -- organizer's object ID is required.\n")
        return

    join_url = input("\nMeeting's Join URL (from its calendar invite): ").strip()
    if not join_url:
        print("Cancelled -- join URL is required.\n")
        return

    print("\nTeams channel webhook URL -- from that channel's \"...\" -> Workflows ->")
    print("\"Send webhook alerts to a channel\" template (see TEAMS_SETUP.md).")
    webhook_url = input("Webhook URL: ").strip()
    if not webhook_url:
        print("Cancelled -- webhook URL is required.\n")
        return

    days_text = input("\nWhich days does it recur? (e.g. 'weekdays', 'daily', 'mon,wed,fri'): ").strip()
    cron_dow, error = _parse_days_to_cron(days_text)
    if error:
        print(error + "\n")
        return

    end_time_text = input("\nWhat time does the meeting usually END? (e.g. 11:00 or 11:00 AM): ").strip()
    hour, minute, error = _parse_time(end_time_text)
    if error:
        print(error + "\n")
        return

    buffer_text = input("\nBuffer in minutes before checking, to let Teams finish "
                         "processing the transcript [default 20]: ").strip()
    try:
        buffer_minutes = int(buffer_text) if buffer_text else 20
    except ValueError:
        print(f"'{buffer_text}' isn't a number -- cancelled.\n")
        return

    check_dt = datetime(2000, 1, 1, hour, minute) + timedelta(minutes=buffer_minutes)

    meetings.append({
        "name": name,
        "organizer_user_id": organizer_user_id,
        "join_url": join_url,
        "webhook_url": webhook_url,
    })
    with open(MEETINGS_CONFIG_FILE, "w") as f:
        json.dump(meetings, f, indent=2)
    print(f"\nSaved to {MEETINGS_CONFIG_FILE}.")

    cron_tag = f"# auto_mom:{name}"
    utc_dow, utc_hour, utc_minute = _local_to_utc_cron(cron_dow, check_dt.hour, check_dt.minute)

    # --- GitHub Actions path (recommended: doesn't need any laptop on) ---
    print(f"\nFor the GitHub Actions pipeline (checks at {check_dt.strftime('%H:%M')} local "
          f"= {utc_hour:02d}:{utc_minute:02d} UTC), add this to .github/workflows/auto_mom.yml's "
          f"`schedule:` list:")
    print(f'  - cron: "{utc_minute} {utc_hour} * * {utc_dow}"  {cron_tag}\n')
    print("Then commit and push that file to main -- schedule triggers only fire from the "
          "default branch. Also update the MEETINGS_CONFIG_JSON repository secret with the "
          f"full current contents of {MEETINGS_CONFIG_FILE} (paste the whole file, not just "
          "this one entry) -- it's a separate copy the workflow reads, not this local file, "
          "so this step is easy to forget.\n")

    # --- Local crontab path (only if this machine will reliably be on) ---
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_dir, "run_auto_mom.sh")
    log_path = os.path.join(repo_dir, "auto_mom.log")
    local_cron_line = (f"{check_dt.minute} {check_dt.hour} * * {cron_dow} "
                        f"{script_path} >> {log_path} 2>&1 {cron_tag}")

    print("Alternatively, a local crontab entry on this machine (only reliable if this "
          "machine is reliably on/awake at check time -- see the GitHub Actions path above "
          "for why that's the recommended one):")
    print(f"  {local_cron_line}\n")
    if not os.path.exists(script_path):
        print(f"Note: {script_path} doesn't exist -- this local-crontab option only works "
              f"when running from a clone of the repo (not the Homebrew-installed rq-agent), "
              f"since it needs run_auto_mom.sh alongside it. The GitHub Actions path above "
              f"doesn't have this requirement.\n")

    install = input("Add the local crontab line now anyway? (y/N): ").strip().lower()
    if install not in ("y", "yes"):
        print("Not installed.\n")
        return

    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    except FileNotFoundError:
        print("`crontab` isn't available on this system -- add the line above manually.\n")
        return
    existing_lines = existing.stdout.splitlines() if existing.returncode == 0 else []

    # Replace any earlier line for this same meeting name rather than piling
    # up duplicates each time /addcall is re-run for it.
    kept_lines = [line for line in existing_lines if cron_tag not in line]
    kept_lines.append(local_cron_line)
    result = subprocess.run(["crontab", "-"], input="\n".join(kept_lines) + "\n", text=True)
    if result.returncode == 0:
        print("Installed -- `crontab -l` will show it.\n")
    else:
        print("Couldn't update crontab -- add the line above manually with `crontab -e`.\n")

    if not os.path.exists(os.path.expanduser("~/.agent_teams_env")):
        print("Reminder: ~/.agent_teams_env doesn't exist yet, so this won't actually run "
              "until TEAMS_TENANT_ID/TEAMS_CLIENT_ID/TEAMS_CLIENT_SECRET are set there "
              "(see run_auto_mom.sh's docstring).\n")


def _workflow_path():
    """Best-effort path to .github/workflows/auto_mom.yml -- only resolves
    to something real when running from a repo checkout (see the same
    caveat in add_scheduled_call() re: the Homebrew-installed rq-agent
    having no such directory at all)."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_dir, ".github", "workflows", "auto_mom.yml")


def _redact(value: str, keep: int = 50) -> str:
    return value if len(value) <= keep else value[:keep] + "..."


def list_scheduled_calls():
    """/listcalls: shows every meeting in meetings_config.json, plus
    whether each one currently has a matching local crontab entry and/or a
    matching entry in the GitHub Actions workflow's schedule: list (both
    tagged '# auto_mom:<name>', same convention add_scheduled_call() writes)
    -- so it's easy to spot a meeting that's configured but not actually
    scheduled anywhere yet."""
    if not os.path.exists(MEETINGS_CONFIG_FILE):
        print(f"No calls configured yet ({MEETINGS_CONFIG_FILE} doesn't exist). "
              f"Use /addcall to add one.\n")
        return
    with open(MEETINGS_CONFIG_FILE) as f:
        meetings = json.load(f)
    if not meetings:
        print("No calls configured yet. Use /addcall to add one.\n")
        return

    try:
        crontab_out = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        crontab_out = ""

    workflow_text = ""
    workflow_path = _workflow_path()
    if os.path.exists(workflow_path):
        with open(workflow_path) as f:
            workflow_text = f.read()

    print(f"{len(meetings)} call(s) configured:\n")
    for m in meetings:
        tag = f"# auto_mom:{m.get('name')}"
        in_cron = tag in crontab_out
        in_workflow = tag in workflow_text
        print(f"- {m.get('name')}")
        print(f"    Organizer:  {m.get('organizer_user_id')}")
        print(f"    Join URL:   {m.get('join_url')}")
        print(f"    Webhook:    {_redact(m.get('webhook_url', ''))}")
        print(f"    Scheduled:  local crontab: {'yes' if in_cron else 'no'}"
              f" | GitHub Actions workflow: {'yes' if in_workflow else 'no'}")
        print()


def remove_scheduled_call(user_input: str):
    """/removecall <name>: removes one meeting from meetings_config.json and
    offers to remove its matching local crontab line. Can't safely
    auto-edit-and-push the GitHub Actions workflow or the MEETINGS_CONFIG_JSON
    secret for the same reasons add_scheduled_call() doesn't -- prints what
    to clean up there instead."""
    match = REMOVECALL_RE.match(user_input)
    name = match.group(1).strip() if match else ""
    if not name:
        print("Usage: /removecall <name>\n")
        return

    if not os.path.exists(MEETINGS_CONFIG_FILE):
        print(f"No calls configured yet ({MEETINGS_CONFIG_FILE} doesn't exist).\n")
        return
    with open(MEETINGS_CONFIG_FILE) as f:
        meetings = json.load(f)
    if not any(m.get("name") == name for m in meetings):
        print(f"No call named '{name}' found. Use /listcalls to see what's configured.\n")
        return

    confirm = input(f"Remove '{name}' from {MEETINGS_CONFIG_FILE}? (y/N): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled.\n")
        return

    meetings = [m for m in meetings if m.get("name") != name]
    with open(MEETINGS_CONFIG_FILE, "w") as f:
        json.dump(meetings, f, indent=2)
    print(f"Removed from {MEETINGS_CONFIG_FILE}.\n")

    cron_tag = f"# auto_mom:{name}"
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        existing_lines = existing.stdout.splitlines() if existing.returncode == 0 else []
    except FileNotFoundError:
        existing_lines = []

    if any(cron_tag in line for line in existing_lines):
        remove_local = input(f"A local crontab entry for '{name}' exists -- remove it too? "
                              f"(y/N): ").strip().lower()
        if remove_local in ("y", "yes"):
            kept_lines = [line for line in existing_lines if cron_tag not in line]
            result = subprocess.run(["crontab", "-"], input="\n".join(kept_lines) + "\n", text=True)
            if result.returncode == 0:
                print("Removed from crontab.\n")
            else:
                print(f"Couldn't update crontab -- remove the line tagged '{cron_tag}' "
                      f"manually with `crontab -e`.\n")

    workflow_path = _workflow_path()
    if os.path.exists(workflow_path):
        with open(workflow_path) as f:
            has_workflow_entry = cron_tag in f.read()
        if has_workflow_entry:
            print(f"Reminder: also remove the line tagged '{cron_tag}' from "
                  f"{workflow_path}'s schedule: list, then commit and push to main.\n")

    print("Reminder: also update the MEETINGS_CONFIG_JSON repository secret with the "
          f"current contents of {MEETINGS_CONFIG_FILE} (now missing '{name}') -- it's a "
          f"separate copy the GitHub Actions workflow reads.\n")


def main():
    pat = os.environ.get("AZURE_DEVOPS_PAT")

    # Seeded from ado_common's constants so an existing setup still gets a
    # one-keystroke confirm instead of being forced to retype it, while a
    # fresh setup for a different org/project just gets asked directly.
    org_state = {"org": DEFAULT_ORG, "project": DEFAULT_PROJECT}
    system_prompt = BASE_SYSTEM_PROMPT
    if pat:
        prompt_org_project(org_state)
        examples = fetch_reference_examples(org_state["org"], org_state["project"], pat)
        if examples:
            system_prompt += "\n\n" + examples
    else:
        print("AZURE_DEVOPS_PAT is not set — /load, /estimate, and /dependency will be "
              "unavailable until you export it, but /mom and general questions still work.\n")
    corrections_context = build_corrections_context(load_corrections())
    if corrections_context:
        system_prompt += "\n\n" + corrections_context

    messages = [{"role": "system", "content": system_prompt}]
    loaded_ids = set()
    current_id = None
    repo_state = {role: {"url": None, "branch": None} for role in REPO_ROLES}

    print(f"{AGENT_NAME} — local dev-team assistant (model: {DEFAULT_MODEL})")
    print("Type /skills to see available commands, or exit to quit.\n")
    if len(sys.argv) > 1 and require_pat(pat):
        current_id = load_item(messages, sys.argv[1], pat, loaded_ids, org_state) or current_id

    # sentence=True matches the completer against the whole line typed so far
    # (not just the current word), so typing "/" shows every skill, and
    # typing "/set " narrows to "/set repo"/"/set org" rather than treating
    # "/set" and "repo" as separate words to complete independently.
    # Only used on a real terminal -- PromptSession doesn't degrade cleanly
    # for piped/non-tty input (garbled output, dropped lines), so scripted
    # or redirected input falls back to plain input() same as before.
    session = (PromptSession(completer=WordCompleter(SKILL_COMMANDS, sentence=True, ignore_case=True))
               if _PROMPT_TOOLKIT_AVAILABLE and sys.stdin.isatty() else None)

    while True:
        try:
            user_input = (session.prompt(">>> ") if session else input(">>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        try:
            if SKILLS_RE.match(user_input):
                print(SKILLS_TEXT + "\n")
                continue

            if SET_REPO_RE.match(user_input):
                for role in REPO_ROLES:
                    repo_state[role] = {"url": None, "branch": None}
                print("Repos cleared — you'll be prompted again on the next /estimate.\n")
                continue

            if SET_ORG_RE.match(user_input):
                org_state["org"] = None
                org_state["project"] = None
                prompt_org_project(org_state)
                print()
                continue

            if MOM_RE.match(user_input):
                generate_mom(messages, user_input)
                continue

            if ADDCALL_RE.match(user_input):
                add_scheduled_call(user_input)
                continue

            if LISTCALLS_RE.match(user_input):
                list_scheduled_calls()
                continue

            if REMOVECALL_RE.match(user_input):
                remove_scheduled_call(user_input)
                continue

            if ESTIMATE_RE.match(user_input):
                if require_pat(pat):
                    estimate_with_codebase(messages, user_input, pat, loaded_ids, repo_state, org_state)
                continue

            if DEPENDENCY_RE.match(user_input):
                dep_ids = ID_RE.findall(user_input)
                if len(dep_ids) < 2:
                    print("Usage: /dependency <id> and <id>\n")
                elif require_pat(pat):
                    analyse_dependency(messages, dep_ids[0], dep_ids[1], pat, loaded_ids, org_state)
                continue

            if LOAD_RE.match(user_input):
                if ID_RE.search(user_input):
                    if require_pat(pat):
                        current_id = load_item(messages, user_input, pat, loaded_ids, org_state) or current_id
                else:
                    print("Usage: /load <id>\n")
                continue

            correct_match = CORRECT_RE.match(user_input)
            if correct_match:
                note = correct_match.group(1).strip()
                if not note:
                    print("Usage: /correct <what was wrong and what it should have been>\n")
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
