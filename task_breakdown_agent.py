#!/usr/bin/env python3
"""Local task breakdown & estimation agent, backed by Ollama.

Usage:
  python3 task_breakdown_agent.py "Add OAuth login to the web app"
  python3 task_breakdown_agent.py --file ticket.md
  cat ticket.md | python3 task_breakdown_agent.py
"""

import argparse
import json
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:3b"

SYSTEM_PROMPT = """You are a senior engineering lead breaking a feature or ticket \
description into an implementation plan for a software team.

Respond with ONLY a single JSON object (no markdown fences, no commentary) matching \
exactly this schema:

{
  "summary": "one-sentence restatement of the goal",
  "assumptions": ["assumption 1", "assumption 2"],
  "tasks": [
    {
      "title": "short task name",
      "description": "what needs to be done, specific enough to hand to a developer",
      "complexity": "S | M | L",
      "estimate_hours_min": number,
      "estimate_hours_max": number,
      "dependencies": ["title of another task, if any"],
      "risks": "key risk or unknown, or empty string"
    }
  ]
}

Rules:
- Break the work into 4-10 concrete, independently workable tasks.
- Order tasks so dependencies come before dependents.
- Estimates are for one mid-level engineer, in hours.
- Complexity: S = under 4h, M = 4-16h, L = over 16h.
- If the description is vague, state your assumptions rather than asking questions.
"""


def call_ollama(description: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        sys.exit(
            f"Could not reach Ollama at {OLLAMA_URL} ({exc}).\n"
            f"Is it running? Try: brew services start ollama"
        )
    content = body["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        sys.exit(f"Model did not return valid JSON:\n\n{content}")


def render(plan: dict) -> str:
    lines = []
    lines.append(f"Summary: {plan.get('summary', '')}")
    assumptions = plan.get("assumptions") or []
    if assumptions:
        lines.append("\nAssumptions:")
        for a in assumptions:
            lines.append(f"  - {a}")

    tasks = plan.get("tasks", [])
    lines.append(f"\nTasks ({len(tasks)}):\n")
    total_min = total_max = 0
    for i, t in enumerate(tasks, 1):
        emin = t.get("estimate_hours_min", 0)
        emax = t.get("estimate_hours_max", 0)
        total_min += emin
        total_max += emax
        lines.append(f"{i}. [{t.get('complexity', '?')}] {t.get('title', '(untitled)')}"
                      f"  ({emin}-{emax}h)")
        lines.append(f"   {t.get('description', '')}")
        deps = t.get("dependencies") or []
        if deps:
            lines.append(f"   depends on: {', '.join(deps)}")
        risks = t.get("risks")
        if risks:
            lines.append(f"   risk: {risks}")
        lines.append("")

    lines.append(f"Total estimate: {total_min}-{total_max} hours "
                  f"(~{total_min/8:.1f}-{total_max/8:.1f} days)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", nargs="?", help="Feature/ticket description")
    parser.add_argument("--file", help="Read description from a file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted text")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            description = f.read()
    elif args.description:
        description = args.description
    elif not sys.stdin.isatty():
        description = sys.stdin.read()
    else:
        parser.error("Provide a description argument, --file, or pipe text via stdin.")

    plan = call_ollama(description, args.model)

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(render(plan))


if __name__ == "__main__":
    main()
