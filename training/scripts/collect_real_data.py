#!/usr/bin/env python3
"""Pulls real (User Story -> real task breakdown) training pairs from Azure DevOps.

Only keeps User Stories that have at least one real child Task (System.LinkTypes.
Hierarchy-Forward) — bare estimate/actual Task rows with no story context are
intentionally excluded (see training plan: not enough context to justify a full
breakdown, would just teach hallucinated structure).

Category (Frontend/Backend/Testing/Other) isn't a real ADO field here, so it's
inferred from task title keywords — an approximation, not ground truth, and is
labeled as such in the output.

Usage:
  export AZURE_DEVOPS_PAT="..."
  python3 training/scripts/collect_real_data.py
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ado_common import (
    DEFAULT_ORG,
    DEFAULT_PROJECT,
    _auth_header,
    _get,
    fetch_work_item,
    fetch_related_context,
    build_context_message,
    strip_html,
)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "real_breakdown_examples.json")

CATEGORY_KEYWORDS = {
    "Testing": ["test", "tc", "qa", "verification", "validation test"],
    "Backend": ["api", "backend", "endpoint", "service", "database", "db", "server",
                "integration", "auth", "firebase", "sentry", "monitoring"],
    "Frontend": ["ui", "screen", "page", "form", "frontend", "design", "layout",
                 "component", "animation", "navigation"],
}


def infer_category(title: str) -> str:
    t = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keywords):
            return category
    return "Other"


def estimate_to_complexity(hours):
    if hours is None:
        return "M"
    if hours < 4:
        return "S"
    if hours <= 16:
        return "M"
    return "L"


def wiql(org, project, pat, query):
    url = f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}/_apis/wit/wiql?api-version=7.1"
    req = urllib.request.Request(url, data=json.dumps({"query": query}).encode(),
                                  headers={**_auth_header(pat), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    pat = os.environ.get("AZURE_DEVOPS_PAT")
    if not pat:
        sys.exit("AZURE_DEVOPS_PAT is not set.")

    org, project = DEFAULT_ORG, DEFAULT_PROJECT

    r = wiql(org, project, pat,
             "SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = @project "
             "AND [System.WorkItemType] = 'User Story'")
    story_ids = [str(w["id"]) for w in r.get("workItems", [])]
    print(f"Found {len(story_ids)} User Stories. Checking each for real child Tasks...")

    examples = []
    for i, story_id in enumerate(story_ids, 1):
        work_item, error = fetch_work_item(org, project, story_id, pat)
        if work_item is None:
            continue
        relations = work_item.get("relations", []) or []
        child_ids = []
        for rel in relations:
            if rel.get("rel") == "System.LinkTypes.Hierarchy-Forward":
                m = re.search(r"/workItems/(\d+)$", rel.get("url", ""))
                if m:
                    child_ids.append(m.group(1))
        if not child_ids:
            continue

        fields = "System.Title,System.WorkItemType,Microsoft.VSTS.Scheduling.OriginalEstimate,Microsoft.VSTS.Scheduling.CompletedWork"
        url = (f"https://dev.azure.com/{urllib.parse.quote(org)}/{urllib.parse.quote(project)}"
               f"/_apis/wit/workitems?ids={','.join(child_ids)}&fields={fields}&api-version=7.1")
        children = _get(url, pat).get("value", [])
        tasks = [c for c in children if c.get("fields", {}).get("System.WorkItemType") == "Task"]
        if not tasks:
            continue

        breakdown_tasks = []
        for t in tasks:
            f = t.get("fields", {})
            title = f.get("System.Title", "")
            if not title:
                continue
            est = f.get("Microsoft.VSTS.Scheduling.OriginalEstimate")
            breakdown_tasks.append({
                "title": title,
                "category": infer_category(title),
                "complexity": estimate_to_complexity(est),
                "estimate_hours_min": max((est or 4) - 1, 0.5) if est else 2,
                "estimate_hours_max": (est or 4) + 1 if est else 6,
            })

        if not breakdown_tasks:
            continue

        related = fetch_related_context(org, project, work_item, pat)
        context = build_context_message(work_item, story_id, related)
        title = work_item.get("fields", {}).get("System.Title", "")

        examples.append({
            "item_id": story_id,
            "title": title,
            "context": context,
            "tasks": breakdown_tasks,
        })
        print(f"  [{i}/{len(story_ids)}] #{story_id}: {len(breakdown_tasks)} real child task(s) -> kept")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved {len(examples)} real (story -> breakdown) examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
