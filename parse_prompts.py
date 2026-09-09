"""
Parse the 25 hand-authored prompts in 25_prompts_five_modes.md into
handwritten/prompts.csv.

The prompts are FINAL — this script transcribes, it never edits. Each data
entry in the markdown is exactly two lines:

    **IF-1** · context · interaction · framing · taxon · salience
    > full prompt text on a single line

Everything else in the file (rationale sections, hardness predictions,
tables) is ignored.

Usage:
    python parse_prompts.py
"""

import csv
import os
import re
import sys

SOURCE_MD = "25_prompts_five_modes.md"
OUT_DIR = "handwritten"
OUT_CSV = os.path.join(OUT_DIR, "prompts.csv")

GENRE_BY_PREFIX = {
    "IF": "instrumental_framing",
    "HH": "higher_harm_default",
    "CA": "consistency_asymmetry",
    "UH": "unverified_harm_dismissal",
    "ED": "edgy_demeaning_content",
}

EXPECTED_TOTAL = 25
EXPECTED_PER_GENRE = 5

# Line 1 of an entry: **ID** · five metadata values separated by U+00B7.
_HEADER = re.compile(
    r"^\*\*([A-Z]{2}-\d+)\*\*\s*·\s*"
    r"([a-z_]+)\s*·\s*([a-z_]+)\s*·\s*([a-z_]+)\s*·\s*([a-z_]+)\s*·\s*([a-z_]+)\s*$"
)


def parse(md_path: str = SOURCE_MD) -> list[dict]:
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rows = []
    for i, line in enumerate(lines):
        m = _HEADER.match(line.strip())
        if not m:
            continue
        source_id, context, interaction, framing, taxon, salience = m.groups()
        prefix = source_id.split("-")[0]
        if prefix not in GENRE_BY_PREFIX:
            print(f"WARNING: unknown id prefix {source_id!r} at line {i + 1}; skipping")
            continue
        # The prompt is the next non-empty line and must be a blockquote.
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not lines[j].startswith("> "):
            print(f"WARNING: {source_id} has no blockquote prompt line; skipping")
            continue
        prompt = lines[j][2:]  # strip the leading "> " and NOTHING else
        rows.append({
            "id": source_id,
            "genre": GENRE_BY_PREFIX[prefix],
            "prompt": prompt,
            "context": context,
            "interaction": interaction,
            "framing": framing,
            "taxon": taxon,
            "salience": salience,
        })
    return rows


if __name__ == "__main__":
    rows = parse()
    by_genre = {}
    for r in rows:
        by_genre[r["genre"]] = by_genre.get(r["genre"], 0) + 1

    ok = len(rows) == EXPECTED_TOTAL and all(
        by_genre.get(g, 0) == EXPECTED_PER_GENRE for g in GENRE_BY_PREFIX.values()
    )
    print(f"Parsed {len(rows)} rows; by genre: {by_genre}")
    if not ok:
        print(f"STOP: expected {EXPECTED_TOTAL} rows, "
              f"{EXPECTED_PER_GENRE} per genre. Not writing output.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}\n")

    print(f"{'id':6s} {'genre':28s} {'words':>5s}  prompt")
    for r in rows:
        wc = len(r["prompt"].split())
        print(f"{r['id']:6s} {r['genre']:28s} {wc:5d}  {r['prompt'][:60]!r}")
