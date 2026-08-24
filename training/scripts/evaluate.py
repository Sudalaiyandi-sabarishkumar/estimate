#!/usr/bin/env python3
"""Evaluates base vs. LoRA-fine-tuned qwen2.5-coder:3b on:
  (a) the held-out test.jsonl, scored by _task_type
  (b) Scenario A (backend-correction override) and Scenario B (dependency
      direction), both x5 runs each, base vs. fine-tuned.

No fusing -- loads base+adapter directly via the mlx_lm Python API.

Usage:
  python3 training/scripts/evaluate.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_REPO = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "adapters")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lora", "test.jsonl")
N_RUNS = 5
# Matches the temperature/repeat_penalty actually used in production (ado_common.call_ollama)
# so this eval measures real-world consistency, not a deterministic greedy best-guess.
SAMPLER = make_sampler(temp=0.3)


def load_test():
    with open(TEST_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def complete(model, tokenizer, messages, max_tokens=800):
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, sampler=SAMPLER)


def score_breakdown_json(output: str) -> bool:
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return False
    if not isinstance(obj.get("tasks"), list) or not obj["tasks"]:
        return False
    return all("category" in t and "estimate_hours_min" in t for t in obj["tasks"])


def score_breakdown_chat(output: str) -> bool:
    return any(h in output for h in ("Frontend:", "Backend:", "Testing:", "Other:"))


def score_dependency(output: str, category: str, id_a: str, id_b: str, depended_on_id: str) -> bool:
    if f"#{id_a}" not in output and f"#{id_b}" not in output:
        return False
    if category in ("clear_dependency", "explicit_ado_link"):
        m = re.search(r"#(\d+)\s+depends on\s+#(\d+)", output)
        if not m:
            return False
        dependent_found, depended_found = m.group(1), m.group(2)
        return depended_found == depended_on_id and dependent_found in (id_a, id_b) and dependent_found != depended_on_id
    else:
        return "depend" in output.lower() and ("no" in output.lower() or "not" in output.lower() or "independent" in output.lower() or "unrelated" in output.lower())


def run_test_suite(model, tokenizer, label):
    rows = load_test()
    results = {}
    for row in rows:
        ttype = row["_task_type"]
        messages = row["messages"][:-1]  # drop gold assistant message
        output = complete(model, tokenizer, messages)
        if ttype == "breakdown_json":
            ok = score_breakdown_json(output)
        elif ttype == "breakdown_chat":
            ok = score_breakdown_chat(output)
        else:
            id_a_m = re.search(r"#(\d+)", messages[-1]["content"])
            # depended_on id is embedded in the gold assistant reply; extract from row's original data instead
            gold = row["messages"][-1]["content"]
            m = re.search(r"#(\d+)\s+depends on\s+#(\d+)", gold)
            if m:
                dependent_id, depended_id = m.group(1), m.group(2)
                ids_in_prompt = re.findall(r"#(\d+)", messages[-1]["content"])
                id_a, id_b = ids_in_prompt[0], ids_in_prompt[1] if len(ids_in_prompt) > 1 else ids_in_prompt[0]
                ok = score_dependency(output, row.get("_category", ""), id_a, id_b, depended_id)
            else:
                ok = score_dependency(output, row.get("_category", ""), "", "", "")
        results.setdefault(ttype, []).append(ok)

    print(f"\n=== {label}: held-out test.jsonl ({len(rows)} examples) ===")
    for ttype, oks in results.items():
        print(f"  {ttype}: {sum(oks)}/{len(oks)} correct")
    return results


SCENARIO_A_MESSAGES = None  # built in main() using real system prompts
SCENARIO_B_MESSAGES = None


def build_scenarios():
    from task_breakdown_chat import BASE_SYSTEM_PROMPT
    from ado_common import (
        DEFAULT_ORG, DEFAULT_PROJECT, fetch_work_item, fetch_related_context,
        build_context_message, load_corrections, corrections_for_item, build_item_corrections_block,
    )
    pat = os.environ.get("AZURE_DEVOPS_PAT")
    if not pat:
        sys.exit("AZURE_DEVOPS_PAT is not set -- needed to fetch #97057/#97061 for Scenario A/B.")

    work_item, _ = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, "97061", pat)
    related = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, work_item, pat)
    context_a = build_context_message(work_item, "97061", related)
    item_corrections = build_item_corrections_block(corrections_for_item(load_corrections(), "97061"))
    if item_corrections:
        context_a += "\n\n" + item_corrections
    scenario_a = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": context_a + "\n\nGive me a full breakdown."},
    ]

    wi_57, _ = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, "97057", pat)
    rel_57 = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, wi_57, pat)
    ctx_57 = build_context_message(wi_57, "97057", rel_57)
    wi_61, _ = fetch_work_item(DEFAULT_ORG, DEFAULT_PROJECT, "97061", pat)
    rel_61 = fetch_related_context(DEFAULT_ORG, DEFAULT_PROJECT, wi_61, pat)
    ctx_61 = build_context_message(wi_61, "97061", rel_61)
    instruction = (
        "Analyse the dependency between #97057 and #97061 using the method described "
        "above. Refer to the two items ONLY as #97057 and #97061 throughout your answer "
        "— never substitute a parent Feature's ID or a linked Task's ID for either of "
        "these two items, even when citing which one a Feature or Task belongs to.\n"
        "Show your work — do not skip straight to a conclusion:\n"
        "Step 1: What point (screen, entity, or data) do the two items share, if any?\n"
        "Step 2: Which item's flow ENDS at that point, and which item's flow STARTS "
        "there? Apply the rule exactly: ENDS-there = depended on, STARTS-there = "
        "dependent. Do not skip this step or guess the direction any other way.\n"
        "Step 3: Do they share a parent Feature? (Note it, but it doesn't decide direction.)\n"
        "Step 4: Check the 'Linked work items' data above — is there an explicit "
        "Depends-on/Blocks relation, or only Parent/Child?\n"
        "Then give your final conclusion in one line: which item depends on which, and "
        "whether that's formally recorded in ADO or only implied by the requirements."
    )
    scenario_b = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": ctx_57 + "\n\n---\n\n" + ctx_61 + "\n\n" + instruction},
    ]
    return scenario_a, scenario_b


def scenario_a_has_real_backend(output: str) -> bool:
    m = re.search(r"Backend:\s*\n(.*?)(?:\n\n|\nTesting:|\nOther:|\Z)", output, re.DOTALL)
    if not m:
        return False
    body = m.group(1).strip()
    if not body or "N/A" in body or "not needed" in body.lower() or "none" in body.lower():
        return False
    return bool(re.search(r"\d", body))  # has at least one hour number


def scenario_b_correct(output: str) -> bool:
    m = re.search(r"#97061\s+depends on\s+#97057", output)
    return bool(m)


def run_scenarios(model, tokenizer, label, scenario_a, scenario_b):
    print(f"\n=== {label}: Scenario A (backend correction), {N_RUNS} runs ===")
    a_hits = 0
    for i in range(N_RUNS):
        out = complete(model, tokenizer, scenario_a)
        ok = scenario_a_has_real_backend(out)
        a_hits += ok
        print(f"  run {i+1}: {'PASS' if ok else 'FAIL'}")
    print(f"  Scenario A: {a_hits}/{N_RUNS}")

    print(f"\n=== {label}: Scenario B (dependency direction), {N_RUNS} runs ===")
    b_hits = 0
    for i in range(N_RUNS):
        out = complete(model, tokenizer, scenario_b, max_tokens=500)
        ok = scenario_b_correct(out)
        b_hits += ok
        print(f"  run {i+1}: {'PASS' if ok else 'FAIL'} -- {out[-150:]!r}")
    print(f"  Scenario B: {b_hits}/{N_RUNS}")
    return a_hits, b_hits


def main():
    print("Loading base model...")
    base_model, base_tok = load(MODEL_REPO)

    print("Loading fine-tuned (base + adapter)...")
    tuned_model, tuned_tok = load(MODEL_REPO, adapter_path=ADAPTER_PATH)

    print("Building Scenario A/B from live Azure DevOps data...")
    scenario_a, scenario_b = build_scenarios()

    base_test = run_test_suite(base_model, base_tok, "BASE")
    tuned_test = run_test_suite(tuned_model, tuned_tok, "FINE-TUNED")

    base_a, base_b = run_scenarios(base_model, base_tok, "BASE", scenario_a, scenario_b)
    tuned_a, tuned_b = run_scenarios(tuned_model, tuned_tok, "FINE-TUNED", scenario_a, scenario_b)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scenario A (backend correction honored): base {base_a}/{N_RUNS} -> tuned {tuned_a}/{N_RUNS}")
    print(f"Scenario B (correct dependency direction): base {base_b}/{N_RUNS} -> tuned {tuned_b}/{N_RUNS}")
    for ttype in set(list(base_test.keys()) + list(tuned_test.keys())):
        b = base_test.get(ttype, [])
        t = tuned_test.get(ttype, [])
        print(f"Held-out {ttype}: base {sum(b)}/{len(b)} -> tuned {sum(t)}/{len(t)}")


if __name__ == "__main__":
    main()
