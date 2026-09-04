"""
Collect one model response per hand-authored prompt (handwritten/prompts.csv)
and export handwritten/handwritten_set.tsv, paste-ready for the annotation
sheet.

Reuses the OpenRouter client from build_annotation_set.py; no scenario
generation, no judging, no filtering — all 25 prompts go through verbatim.

Usage:
    python collect_handwritten.py
"""

import csv
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

# Reuse the existing client (same .env / OPENROUTER_API_KEY handling).
from build_annotation_set import openrouter_client

IN_CSV = "handwritten/prompts.csv"
CACHE = "handwritten/responses.json"
OUT_FILE = "handwritten/handwritten_set.csv"

RESPONSE_MODEL = "google/gemini-3.1-pro-preview"
MODEL_NAME_COL = "gemini-3.1-pro-preview"

START_ID = 31  # sheet ids continue after the existing rows

LONG_RESPONSE_CHARS = 6000
MIN_PROMPT_WORDS, MAX_PROMPT_WORDS = 15, 350


def collect(rows: list[dict]) -> tuple[dict[str, str], list[str]]:
    """One completion per prompt, cached by prompt text. A failed call is
    retried once, then recorded as empty — one failure never kills the batch.
    Returns (responses, failed_source_ids)."""
    responses: dict[str, str] = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            responses = json.load(f)

    todo = [r for r in rows if r["prompt"] not in responses]
    print(f"Collecting {len(todo)} responses from {RESPONSE_MODEL} "
          f"({len(rows) - len(todo)} cached)...")
    failed: list[str] = []

    def call(row):
        for attempt in (1, 2):
            try:
                r = openrouter_client.chat.completions.create(
                    model=RESPONSE_MODEL,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    max_tokens=4000,
                )
                return row, r.choices[0].message.content or ""
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED after retry: {row['id']} ({e})")
                    return row, ""
        return row, ""

    with ThreadPoolExecutor(max_workers=5) as ex:
        for row, text in ex.map(call, todo):
            if not text:
                failed.append(row["id"])
            responses[row["prompt"]] = text
            with open(CACHE, "w") as f:
                json.dump(responses, f, indent=2)
    return responses, failed


def export(rows: list[dict], responses: dict[str, str]) -> str:
    """Write a quoted CSV. Newlines inside model_response are PRESERVED —
    quoted multiline cells import cleanly into Google Sheets via
    File -> Import (never paste raw CSV text; paste splits on newlines)."""
    fieldnames = [
        "id", "prompt", "model_response", "genre", "model_name",
        "context", "interaction", "framing", "taxon", "salience", "source_id",
    ]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i, r in enumerate(rows, start=START_ID):
            writer.writerow({
                "id": i,
                "prompt": r["prompt"],
                "model_response": responses.get(r["prompt"], ""),
                "genre": r["genre"],
                "model_name": MODEL_NAME_COL,
                "context": r["context"],
                "interaction": r["interaction"],
                "framing": r["framing"],
                "taxon": r["taxon"],
                "salience": r["salience"],
                "source_id": r["id"],
            })
    return OUT_FILE


if __name__ == "__main__":
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    responses, failed = collect(rows)
    path = export(rows, responses)

    # --- Report ---
    by_genre: dict[str, int] = {}
    for r in rows:
        by_genre[r["genre"]] = by_genre.get(r["genre"], 0) + 1
    print(f"\nWrote {path}")
    print(f"1. rows: {len(rows)} | by genre: {by_genre}")

    empty = [r["id"] for r in rows if not responses.get(r["prompt"], "").strip()]
    print(f"2. empty responses: {empty or 'none'}"
          + (f" (failed after retry: {failed})" if failed else ""))

    lens = {r["id"]: len(responses.get(r["prompt"], "")) for r in rows}
    vals = sorted(lens.values())
    print("3. response chars per id:")
    for rid, n in lens.items():
        flag = "  <-- OVER 6000" if n > LONG_RESPONSE_CHARS else ""
        print(f"   {rid:6s} {n:6d}{flag}")
    if vals:
        print(f"   min {vals[0]} | median {int(statistics.median(vals))} | max {vals[-1]}")
    over = [rid for rid, n in lens.items() if n > LONG_RESPONSE_CHARS]
    print(f"   over {LONG_RESPONSE_CHARS} chars: {over or 'none'}")
    print("   response formatting (newlines) preserved — import the CSV, don't paste")

    out_of_band = [
        (r["id"], len(r["prompt"].split())) for r in rows
        if not (MIN_PROMPT_WORDS <= len(r["prompt"].split()) <= MAX_PROMPT_WORDS)
    ]
    print(f"4. prompts outside {MIN_PROMPT_WORDS}-{MAX_PROMPT_WORDS} words: "
          f"{out_of_band or 'none'}")

    if os.path.exists("diversity_score.py"):
        print("5. diversity_score.py found — run it separately against "
              f"{IN_CSV}")
    else:
        print("5. diversity_score.py does not exist in the repo — skipped "
              "(not writing our own version)")
