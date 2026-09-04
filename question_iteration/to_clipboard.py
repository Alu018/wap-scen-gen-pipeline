"""
Turn a question_iteration run JSON into paste-ready quoted TSV and put it on
the macOS clipboard.

Quoted TSV is the format Google Sheets itself uses on the clipboard, so
multi-paragraph responses paste as single tall cells with their line breaks
intact — no File->Import needed. Click a cell, Cmd+V.

Usage:
    python question_iteration/to_clipboard.py question_iteration/run_X.json
    python question_iteration/to_clipboard.py run_X.json --no-copy   # just write the .paste.tsv
"""

import csv
import io
import json
import subprocess
import sys

COLUMNS = ["name", "genre_or_cue", "prompt", "model", "response"]


def to_tsv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([
            r.get("name", ""),
            r.get("genre", r.get("cue", "")),
            r.get("prompt", ""),
            (r.get("model", "") or "").split("/")[-1],
            r.get("response", ""),
        ])
    return buf.getvalue()


if __name__ == "__main__":
    path = sys.argv[1]
    rows = json.load(open(path))
    tsv = to_tsv(rows)
    out = path.replace(".json", ".paste.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(tsv)
    print(f"wrote {out} ({len(rows)} rows)")
    if "--no-copy" not in sys.argv:
        subprocess.run(["pbcopy"], input=tsv.encode("utf-8"), check=True)
        print("ON YOUR CLIPBOARD now — click a cell in Sheets and Cmd+V")
