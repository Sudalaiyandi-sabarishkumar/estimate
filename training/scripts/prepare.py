#!/usr/bin/env python3
"""Pools real + synthetic examples, tags each with _task_type, checks token
length against max_seq_length, and writes a stratified 80/10/10 split to
training/data/lora/{train,valid,test}.jsonl in mlx-lm's chat format.

Usage:
  python3 training/scripts/prepare.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from azure_task_breakdown import BASE_SYSTEM_PROMPT as JSON_SYSTEM_PROMPT
from task_breakdown_chat import BASE_SYSTEM_PROMPT as CHAT_SYSTEM_PROMPT

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SYN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "lora")
MODEL_REPO = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
MAX_SEQ_LEN = 2048

random.seed(0)


def render_real_json(example):
    tasks = []
    for t in example["tasks"]:
        tasks.append({
            "title": t["title"],
            "category": t["category"],
            "description": f"{t['title']} — as scoped in the real ticket's child task.",
            "complexity": t["complexity"],
            "estimate_hours_min": t["estimate_hours_min"],
            "estimate_hours_max": t["estimate_hours_max"],
            "dependencies": [],
            "risks": "",
        })
    assistant_obj = {
        "summary": example["title"][:200],
        "assumptions": ["Category is inferred from task title keywords, not a real ADO field."],
        "tasks": tasks,
    }
    return {
        "messages": [
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {"role": "user", "content": example["context"]},
            {"role": "assistant", "content": json.dumps(assistant_obj)},
        ],
        "_task_type": "breakdown_json",
        "_source": "real",
        "_item_id": example["item_id"],
    }


def render_real_chat(example):
    by_cat = {}
    for t in example["tasks"]:
        by_cat.setdefault(t["category"], []).append(t)
    lines = []
    grand_min = grand_max = 0
    for cat in ["Frontend", "Backend", "Testing", "Other"]:
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"{cat}:")
        cmin = cmax = 0
        for t in items:
            lines.append(f"- {t['title']} ({t['estimate_hours_min']}-{t['estimate_hours_max']}h)")
            cmin += t["estimate_hours_min"]
            cmax += t["estimate_hours_max"]
        lines.append(f"Total: {cmin}-{cmax}h\n")
        grand_min += cmin
        grand_max += cmax
    lines.append(f"Total estimate: {grand_min}-{grand_max} hours (~{grand_min/8:.1f}-{grand_max/8:.1f} days)")
    return {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": example["context"] + "\n\nGive me a full breakdown."},
            {"role": "assistant", "content": "\n".join(lines)},
        ],
        "_task_type": "breakdown_chat",
        "_source": "real",
        "_item_id": example["item_id"],
    }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def strat_key(row):
    if row["_task_type"] == "dependency_reasoning":
        return f"dependency_reasoning:{row.get('_category')}"
    return row["_task_type"]


def stratified_split(rows, train_frac=0.8, valid_frac=0.1):
    buckets = {}
    for r in rows:
        buckets.setdefault(strat_key(r), []).append(r)

    train, valid, test = [], [], []
    for key, bucket in buckets.items():
        random.shuffle(bucket)
        n = len(bucket)
        n_valid = max(1, round(n * valid_frac)) if n >= 3 else 0
        n_test = max(1, round(n * (1 - train_frac - valid_frac))) if n >= 3 else (1 if n >= 2 else 0)
        n_train = n - n_valid - n_test
        train += bucket[:n_train]
        valid += bucket[n_train:n_train + n_valid]
        test += bucket[n_train + n_valid:]
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return train, valid, test


def main():
    real_raw_path = os.path.join(RAW_DIR, "real_breakdown_examples.json")
    real_examples = json.load(open(real_raw_path)) if os.path.exists(real_raw_path) else []

    pool = []
    for ex in real_examples:
        pool.append(render_real_json(ex))
        pool.append(render_real_chat(ex))

    for fname in ["breakdown_json_synthetic.jsonl", "breakdown_chat_synthetic.jsonl", "dependency_synthetic.jsonl"]:
        path = os.path.join(SYN_DIR, fname)
        if os.path.exists(path):
            pool += load_jsonl(path)

    print(f"Pooled {len(pool)} total examples before length check:")
    by_type = {}
    for r in pool:
        by_type[strat_key(r)] = by_type.get(strat_key(r), 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")

    # Token length check against the actual target tokenizer.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_REPO)
        too_long = []
        kept = []
        for r in pool:
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            n_tokens = len(tok(text)["input_ids"])
            if n_tokens > MAX_SEQ_LEN:
                too_long.append((r.get("_item_id", r.get("_task_type")), n_tokens))
            else:
                kept.append(r)
        if too_long:
            print(f"\nDropping {len(too_long)} examples over max_seq_length={MAX_SEQ_LEN}:")
            for name, n in too_long:
                print(f"  {name}: {n} tokens")
        pool = kept
    except Exception as exc:
        print(f"\nWARNING: could not load tokenizer to check lengths ({exc}); skipping length check.")

    train, valid, test = stratified_split(pool)

    os.makedirs(OUT_DIR, exist_ok=True)

    def write_jsonl(name, rows):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}")

    write_jsonl("train.jsonl", train)
    write_jsonl("valid.jsonl", valid)
    write_jsonl("test.jsonl", test)


if __name__ == "__main__":
    main()
