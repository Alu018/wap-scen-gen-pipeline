#!/usr/bin/env python3
"""
Send every prompt in a question-iteration batch to the three benchmark models
and write a run JSON (prompt-major, model order fixed) that json_to_tsv.py
turns into paste-ready TSV.

    python3 question_iteration/run_batch.py question_iteration/batch_X.json [--copy]

Input: JSON list of {label, prompt, context, interaction, framing, taxon,
salience, cue, ...}. Prompts are sent VERBATIM as a single user turn, no
system prompt. Output: question_iteration/run_X.json with one row per
(prompt, model) plus, with --copy, run_X.sheet.tsv on the clipboard.
Responses are cached per (model, prompt) in run_X.json so re-runs only fill
gaps; a call that fails after one retry is recorded as "" and reported.
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_annotation_set import openrouter_client  # noqa: E402  (.env + OPENROUTER_API_KEY)

MODELS = [
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-terra",
]
MAX_TOKENS = 4000


def call(model, prompt):
    for attempt in (1, 2):
        try:
            r = openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                print(f"  FAILED after retry [{model}] {prompt[:40]!r}: {e}")
    return ""


def main(batch_path, copy):
    batch = json.load(open(batch_path, encoding="utf-8"))
    run_path = re.sub(r"batch_", "run_", batch_path, count=1)
    cached = {}
    if os.path.exists(run_path):
        for r in json.load(open(run_path, encoding="utf-8")):
            if r.get("response"):
                cached[(r["model"], r["prompt"])] = r["response"]

    jobs = [(m, b["prompt"]) for b in batch for m in MODELS if (m, b["prompt"]) not in cached]
    print(f"{len(batch)} prompts x {len(MODELS)} models: {len(jobs)} to fetch, {len(cached)} cached")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for (m, p), text in zip(jobs, ex.map(lambda j: call(*j), jobs)):
            cached[(m, p)] = text

    rows, failed = [], []
    for b in batch:
        for m in MODELS:
            text = cached.get((m, b["prompt"]), "")
            if not text:
                failed.append((b.get("label", ""), m))
            row = {k: v for k, v in b.items()}
            row["model"] = m
            row["response"] = text
            rows.append(row)
    json.dump(rows, open(run_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} rows to {run_path}")
    if failed:
        print(f"EMPTY responses: {failed}")

    tsv_path = run_path[:-5] + ".sheet.tsv"
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "json_to_tsv.py"), run_path, tsv_path]
    if copy:
        cmd.append("--copy")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--copy"]
    if not args:
        sys.exit(__doc__.strip())
    main(args[0], copy="--copy" in sys.argv)
