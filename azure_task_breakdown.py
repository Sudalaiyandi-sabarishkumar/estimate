#!/usr/bin/env python3
"""Task breakdown & estimation agent grounded in an Azure DevOps work item.

Fetches a work item's title, description, acceptance criteria, linked items,
attachments, and comments from Azure DevOps, then asks a local Ollama model
to break the work into Frontend, Backend, and Testing tasks with estimates —
calibrated against this team's own past estimate-vs-actual data.

Setup (one-time):
  export AZURE_DEVOPS_PAT="..."   # PAT with Work Items (Read) scope

Usage:
  python3 azure_task_breakdown.py 101490
  python3 azure_task_breakdown.py 101490 --org rootquotient --project "Buddhi Mantra"
  python3 azure_task_breakdown.py 101490 --json
"""

import argparse
import json
import os
import sys

from rq_agent.ado_common import (
    DEFAULT_ORG,
    DEFAULT_PROJECT,
    DEFAULT_MODEL,
    fetch_work_item,
    fetch_related_context,
    fetch_reference_examples,
    build_context_message,
    call_ollama,
)

BASE_SYSTEM_PROMPT = """You are a senior engineering lead breaking a real Azure DevOps \
work item into an implementation plan for a software team.

You will be given the work item's title, type, description, acceptance criteria, \
linked items, attachments, and comments. Base your breakdown on that actual content, \
not generic assumptions.

Respond with ONLY a single JSON object (no markdown fences, no commentary) matching \
exactly this schema:

{
  "summary": "one-sentence restatement of the goal, grounded in the work item",
  "assumptions": ["assumption 1", "assumption 2"],
  "tasks": [
    {
      "title": "short task name",
      "category": "Frontend | Backend | Testing | Other",
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
- Cover Frontend, Backend, and Testing separately; only include "Other" for things
  like design review, documentation, or DevOps work that don't fit those three.
- Break the work into concrete, independently workable tasks (typically 5-12 total).
- Order tasks so dependencies come before dependents.
- Estimates are for one mid-level engineer, in hours.
- Complexity: S = under 4h, M = 4-16h, L = over 16h.
- If the description or acceptance criteria are vague, state assumptions rather
  than asking questions.
"""


def get_plan(description: str, model: str) -> dict:
    print(f"Generating breakdown with {model} (this can take a few minutes)...", file=sys.stderr)
    reply = call_ollama(
        [
            {"role": "system", "content": description["system_prompt"]},
            {"role": "user", "content": description["context"]},
        ],
        model,
        show_progress=False,
    )
    if reply is None:
        sys.exit(1)
    content = reply.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        sys.exit(f"Model did not return valid JSON:\n\n{content}")


def render(plan: dict, item_id: str, item_url: str) -> str:
    lines = [f"Work item: #{item_id}  ({item_url})", f"Summary: {plan.get('summary', '')}"]

    assumptions = plan.get("assumptions") or []
    if assumptions:
        lines.append("\nAssumptions:")
        for a in assumptions:
            lines.append(f"  - {a}")

    tasks = plan.get("tasks", [])
    by_category = {}
    for t in tasks:
        by_category.setdefault(t.get("category", "Other"), []).append(t)

    grand_min = grand_max = 0
    for category in ["Frontend", "Backend", "Testing", "Other"]:
        cat_tasks = by_category.get(category)
        if not cat_tasks:
            continue
        lines.append(f"\n{category}:")
        cat_min = cat_max = 0
        for t in cat_tasks:
            emin = t.get("estimate_hours_min", 0)
            emax = t.get("estimate_hours_max", 0)
            cat_min += emin
            cat_max += emax
            lines.append(f"  [{t.get('complexity', '?')}] {t.get('title', '(untitled)')}"
                          f"  ({emin}-{emax}h)")
            lines.append(f"    {t.get('description', '')}")
            deps = t.get("dependencies") or []
            if deps:
                lines.append(f"    depends on: {', '.join(deps)}")
            risks = t.get("risks")
            if risks:
                lines.append(f"    risk: {risks}")
        lines.append(f"  Subtotal: {cat_min}-{cat_max}h")
        grand_min += cat_min
        grand_max += cat_max

    lines.append(f"\nTotal estimate: {grand_min}-{grand_max} hours "
                  f"(~{grand_min/8:.1f}-{grand_max/8:.1f} days)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("item_id", help="Azure DevOps work item ID")
    parser.add_argument("--org", default=DEFAULT_ORG)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted text")
    args = parser.parse_args()

    pat = os.environ.get("AZURE_DEVOPS_PAT")
    if not pat:
        sys.exit("AZURE_DEVOPS_PAT is not set. Run: export AZURE_DEVOPS_PAT=\"...\"")

    work_item, error = fetch_work_item(args.org, args.project, args.item_id, pat)
    if work_item is None:
        sys.exit(error)
    related = fetch_related_context(args.org, args.project, work_item, pat)
    context = build_context_message(work_item, args.item_id, related)
    item_url = work_item.get("_links", {}).get("html", {}).get("href", "")

    system_prompt = BASE_SYSTEM_PROMPT
    examples = fetch_reference_examples(args.org, args.project, pat)
    if examples:
        system_prompt += "\n\n" + examples

    plan = get_plan({"system_prompt": system_prompt, "context": context}, args.model)

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(render(plan, args.item_id, item_url))


if __name__ == "__main__":
    main()
