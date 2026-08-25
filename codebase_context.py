"""Local-clone + keyword-grep retrieval for codebase-aware estimation.

Clones a stable branch of the actual product repo (Azure Repos), caches it on
disk, and pulls a small, budgeted set of grep-matched snippets to ground the
model — no embeddings, no semantic search.
"""

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess

CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".repo_cache")

CODEBASE_NUM_CTX = 8192  # only used for the estimate command, see task_breakdown_chat.py
CODEBASE_CONTEXT_CHAR_BUDGET = 7000
MAX_FILES = 8
MAX_LINES_PER_FILE = 40

# Prefix marking a "this repo appears unrelated to this ticket" result, so
# callers can detect it programmatically (see build_codebase_context) instead
# of the model being the only thing that might notice and mention it.
UNRELATED_REPO_MARKER = "CODEBASE CONTEXT: none of this ticket's distinctive terms"
ARCHITECTURE_CHAR_BUDGET = 1500

# Common names for an app's backend/API-integration layer, checked as
# case-insensitive directory-name substrings across common stacks (Flutter/
# mobile, web, backend services).
ARCHITECTURE_DIR_HINTS = [
    "api_repository", "api", "repository", "repositories", "service", "services",
    "network", "networking", "remote", "data_source", "datasource", "backend",
]

EXCLUDE_PATHSPECS = [
    ":(exclude)**/node_modules/**", ":(exclude)**/build/**",
    ":(exclude)**/dist/**", ":(exclude)**/.gradle/**",
    ":(exclude)**/vendor/**", ":(exclude)**/*.min.js",
    ":(exclude)**/*.lock", ":(exclude)**/package-lock.json",
    # Binary file types git's own -I flag doesn't reliably catch (e.g. some
    # .ttf files don't trip git's NUL-byte binary heuristic) -- their raw
    # bytes can coincidentally "match" short common words with inflated
    # counts and bury genuinely relevant source files out of the top ranks.
    # Git pathspecs don't support brace-expansion, so each extension is its
    # own pattern rather than a single "*.{a,b,c}" glob.
    ":(exclude)**/*.ttf", ":(exclude)**/*.otf",
    ":(exclude)**/*.woff", ":(exclude)**/*.woff2",
    ":(exclude)**/*.png", ":(exclude)**/*.jpg", ":(exclude)**/*.jpeg",
    ":(exclude)**/*.gif", ":(exclude)**/*.ico", ":(exclude)**/*.webp",
    ":(exclude)**/*.pdf", ":(exclude)**/*.zip", ":(exclude)**/*.jar",
    ":(exclude)**/*.class", ":(exclude)**/*.so", ":(exclude)**/*.dylib",
    ":(exclude)**/*.dll", ":(exclude)**/*.exe", ":(exclude)**/*.mp3",
    ":(exclude)**/*.mp4", ":(exclude)**/*.mov",
]

STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "into", "when",
             "should", "have", "will", "user", "users", "screen", "page", "app",
             "application", "then", "there", "their", "also", "need", "needs",
             "able", "after", "before", "only", "must", "system", "display"}

# Generic CRUD verbs and common UI nouns -- excluded specifically from
# _title_core_terms (not the broader STOPWORDS used for scoring everywhere)
# because they're common enough to appear as real whole-word matches in
# almost any codebase regardless of domain, so they're useless as a signal
# for "is this repo even related to this ticket." A word surviving both
# STOPWORDS and this list is much more likely to be the ticket's actual
# feature-specific noun (e.g. "wishlist", "checkout", "assessment").
GENERIC_ACTION_TERMS = {
    "add", "adds", "adding", "remove", "removes", "removing", "delete", "deletes",
    "update", "updates", "updating", "create", "creates", "creating", "view",
    "views", "viewing", "click", "clicks", "clicking", "save", "saves", "saving",
    "select", "selects", "selecting", "choose", "chooses", "enter", "enters",
    "entering", "show", "shows", "showing", "want", "wants", "wanting", "make",
    "makes", "item", "items", "list", "lists", "option", "options", "detail",
    "details", "button", "buttons", "icon", "icons", "field", "fields", "form",
    "forms", "later", "mobile", "product", "products", "access", "provide",
}

_PHRASE_RE = re.compile(r'"([^"]{3,60})"|\b([A-Z][a-zA-Z0-9]*(?:[ \t]+[A-Z][a-zA-Z0-9]*){1,4})\b')
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")


def _cache_key(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode()).hexdigest()[:16]


def _scoped_header_arg(repo_url: str, pat: str) -> str:
    token = base64.b64encode(f":{pat}".encode()).decode()
    return f"http.{repo_url}.extraheader=AUTHORIZATION: basic {token}"


def clone_or_update_repo(repo_url: str, branch: str, pat: str):
    """Returns (local_path, error). Reuses a shallow single-branch clone across
    runs; re-clones if the branch changed, otherwise does a shallow fetch +
    hard reset. Never writes the PAT into the URL, .git/config, or disk."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    dest = os.path.join(CACHE_ROOT, _cache_key(repo_url))
    meta_path = os.path.join(dest, ".agent_meta.json")
    header = _scoped_header_arg(repo_url, pat)

    if os.path.isdir(os.path.join(dest, ".git")):
        meta = {}
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        if meta.get("branch") == branch:
            r = subprocess.run(
                ["git", "-C", dest, "-c", header, "fetch", "--depth", "1", "origin", branch],
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return None, f"git fetch failed: {r.stderr.strip()[:500]}"
            r2 = subprocess.run(
                ["git", "-C", dest, "reset", "--hard", f"origin/{branch}"],
                capture_output=True, text=True, timeout=30)
            if r2.returncode != 0:
                return None, f"git reset failed: {r2.stderr.strip()[:500]}"
            return dest, None
        shutil.rmtree(dest)  # branch changed -- shallow single-branch clone can't add a branch cheaply

    r = subprocess.run(
        ["git", "-c", header, "clone", "--depth", "1", "--single-branch",
         "--branch", branch, repo_url, dest],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return None, f"git clone failed: {r.stderr.strip()[:800]}"
    with open(meta_path, "w") as f:
        json.dump({"repo_url": repo_url, "branch": branch}, f)
    return dest, None


def extract_keywords(text: str, limit: int = 25) -> list:
    """Returns [(term, weight)] -- weight 2 for quoted/Title-Case phrases
    (likely screen/entity names), weight 1 for generic significant words."""
    terms = {}
    for m in _PHRASE_RE.finditer(text):
        phrase = (m.group(1) or m.group(2)).strip()
        if phrase and phrase.lower() not in STOPWORDS:
            terms[phrase] = 2
    for word in _WORD_RE.findall(text):
        if word.lower() not in STOPWORDS:
            terms.setdefault(word, 1)
    return list(terms.items())[:limit]


def _grep_counts(repo_path: str, term: str) -> dict:
    cmd = ["git", "-C", repo_path, "grep", "-c", "-I", "-i", "-e", term, "--", "."] + EXCLUDE_PATHSPECS
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if r.returncode not in (0, 1):  # 1 = no matches, not a failure
        return {}
    counts = {}
    for line in r.stdout.splitlines():
        path, _, cnt = line.rpartition(":")
        if path:
            counts[path] = int(cnt)
    return counts


def score_files(repo_path: str, weighted_terms: list) -> list:
    """Ranks files by (distinct terms matched, weighted hit count) -- favors
    breadth of relevance over one term repeating in an unrelated file."""
    scores = {}
    for term, weight in weighted_terms:
        for path, cnt in _grep_counts(repo_path, term).items():
            entry = scores.setdefault(path, {"distinct": 0, "weighted": 0, "terms": []})
            entry["distinct"] += 1
            entry["weighted"] += cnt * weight
            entry["terms"].append(term)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1]["distinct"], kv[1]["weighted"]), reverse=True)
    return ranked[:MAX_FILES]


def _file_snippet(repo_path: str, path: str, terms: list) -> str:
    pattern = "|".join(re.escape(t) for t in set(terms))
    cmd = ["git", "-C", repo_path, "grep", "-n", "-A3", "-B3", "-i", "-E", "-e", pattern, "--", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return "\n".join(r.stdout.splitlines()[:MAX_LINES_PER_FILE])


def _title_core_terms(title: str, limit: int = 8) -> list:
    """Longer, non-generic words from the ticket TITLE specifically (not the
    full acceptance criteria, which is mostly generic ADO/Given-When-Then
    boilerplate shared across every ticket). Filtered against both STOPWORDS
    and GENERIC_ACTION_TERMS so common CRUD verbs/UI nouns ('remove', 'item',
    'mobile') don't crowd out the one word that actually names the feature
    (e.g. 'wishlist', 'checkout', 'assessment') -- a generic word will produce
    real whole-word matches in almost any codebase regardless of domain, so
    it's useless as a signal for "is this repo even related to this ticket."""
    excluded = STOPWORDS | GENERIC_ACTION_TERMS
    words = [w for w in _WORD_RE.findall(title) if w.lower() not in excluded and len(w) >= 5]
    seen = []
    for w in words:
        if w.lower() not in {s.lower() for s in seen}:
            seen.append(w)
    return seen[:limit]


def _any_term_matches_anywhere(repo_path: str, terms: list) -> bool:
    for term in terms:
        # -w (whole word) avoids counting a substring hit inside an unrelated
        # identifier (e.g. "remove" inside "removeUnderscore") as a match.
        cmd = ["git", "-C", repo_path, "grep", "-q", "-I", "-i", "-w", "-e", term, "--", "."] + EXCLUDE_PATHSPECS
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        if r.returncode == 0:
            return True
    return False


def build_codebase_context(repo_path: str, work_item_text: str,
                            char_budget: int = CODEBASE_CONTEXT_CHAR_BUDGET,
                            title: str = None) -> str:
    """Orchestrates keyword extraction -> git grep scoring -> snippet pull ->
    char-budget-capped concatenation. Never raises on empty results -- returns
    an explicit 'nothing found' string so the model doesn't hallucinate.

    When title is given, first sanity-checks whether ANY of the ticket's own
    distinctive words appear anywhere in this repo at all. Generic
    acceptance-criteria vocabulary ("icon", "list", "update") coincidentally
    matches something in almost every repo, which without this check can make
    a completely unrelated repo (wrong product, wrong team) look like it has
    "matching files" -- this catches that before it reaches the model."""
    if title:
        core_terms = _title_core_terms(title)
        if core_terms and not _any_term_matches_anywhere(repo_path, core_terms):
            return (f"{UNRELATED_REPO_MARKER} ({', '.join(core_terms)}) appear anywhere in "
                    f"this repo. This repo is likely unrelated to this ticket (wrong repo, or "
                    f"this feature genuinely doesn't touch this part of the stack) -- treat as "
                    f"no evidence, do not guess at file/module names for this repo.")

    weighted_terms = extract_keywords(work_item_text)
    if not weighted_terms:
        return "CODEBASE CONTEXT: couldn't extract meaningful keywords from this work item."
    ranked = score_files(repo_path, weighted_terms)
    if not ranked:
        sample = ", ".join(t for t, _ in weighted_terms[:10])
        return f"CODEBASE CONTEXT: grepped for [{sample}] but found no matching files in this branch."
    header = f"CODEBASE CONTEXT (grepped from the cloned branch, top {len(ranked)} matching files):"
    blocks, total = [header], len(header)
    included_any_file = False
    for i, (path, info) in enumerate(ranked):
        snippet = _file_snippet(repo_path, path, info["terms"])
        block = f"\n--- {path} (matched: {', '.join(sorted(set(info['terms'])))}) ---\n{snippet}"
        remaining = char_budget - total
        if len(block) > remaining:
            if not included_any_file:
                # A real snippet (even the single top-ranked file) exceeding
                # the whole budget must still contribute *something* -- the
                # previous behavior dropped it entirely, leaving the model
                # with zero real evidence and a header claiming matches exist.
                trimmed = block[:max(remaining, 300)]
                blocks.append(trimmed + "\n... (truncated to fit context budget)")
                total += len(trimmed)
                included_any_file = True
            blocks.append(f"\n(truncated -- {len(ranked) - i} more matching files omitted to fit context budget)")
            break
        blocks.append(block)
        total += len(block)
        included_any_file = True
    return "".join(blocks)


def build_architecture_overview(repo_path: str, char_budget: int = ARCHITECTURE_CHAR_BUDGET) -> str:
    """Lists this app's existing backend/API-integration layer (if any), so the
    model can judge whether a brand-new feature needs backend work based on
    established patterns in this codebase -- not just keyword-grepped noise,
    which finds nothing useful for a feature that doesn't exist yet (a new
    feature's ticket vocabulary won't match any of its own not-yet-written
    backend code, so build_codebase_context alone is blind to this)."""
    found_dirs = []
    for hint in ARCHITECTURE_DIR_HINTS:
        r = subprocess.run(
            ["find", repo_path, "-type", "d", "-iname", f"*{hint}*", "-not", "-path", "*/.git/*"],
            capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            rel = os.path.relpath(line, repo_path)
            if rel not in found_dirs:
                found_dirs.append(rel)

    if not found_dirs:
        return "ARCHITECTURE OVERVIEW: no dedicated API/service/repository layer found in this repo."

    lines = ["ARCHITECTURE OVERVIEW (this app's existing backend/API-integration layer, "
             "for judging whether a NEW feature needs backend work -- if this app already "
             "talks to a backend for other features, a new feature involving persisted or "
             "account-tied data likely needs to as well, even with no existing code for it):"]
    total = len(lines[0])
    sample_file = None
    for d in found_dirs[:3]:
        full_dir = os.path.join(repo_path, d)
        try:
            files = sorted(os.listdir(full_dir))[:10]
        except OSError:
            continue
        block = f"\n{d}/: " + ", ".join(files)
        if total + len(block) > char_budget:
            break
        lines.append(block)
        total += len(block)
        if sample_file is None:
            for f in files:
                if f.endswith((".dart", ".ts", ".js", ".py", ".java", ".kt", ".swift")):
                    sample_file = os.path.join(full_dir, f)
                    break

    if sample_file:
        try:
            with open(sample_file, errors="ignore") as f:
                snippet = "".join(f.readlines()[:15])
            rel = os.path.relpath(sample_file, repo_path)
            block = f"\n\nExample ({rel}):\n{snippet}"
            if total + len(block) <= char_budget:
                lines.append(block)
        except OSError:
            pass

    return "".join(lines)
