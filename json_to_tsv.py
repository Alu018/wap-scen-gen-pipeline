#!/usr/bin/env python3
"""
Convert a question-iteration run (JSON list of {name, prompt, model, response,
...}) into a quoted TSV for Google Sheets.

    python3 json_to_tsv.py run.json [out.tsv] [--copy]

Quoted TSV pastes cleanly into Sheets (multiline cells stay in one cell), so
with --copy it lands on the clipboard: click a cell, Cmd+V. File > Import >
Upload (separator = Tab) works as the fallback.

Responses are markdown-stripped for spreadsheet readability (line breaks
kept). Prompts are NEVER altered beyond tab/CRLF safety — the register and
typos are part of the benchmark design.
"""
import json
import csv
import re
import subprocess
import sys

COLS = ["id", "prompt", "model_response", "model_name",
        "context", "interaction", "framing", "taxon", "salience",
        "cue", "score (1 is good, 0 is bad)"]


def to_plain(s):
    """Strip markdown so the cell is readable in a spreadsheet. Keeps line breaks."""
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.M)      # headings
    s = re.sub(r"^\s*[-*_]{3,}\s*$", "", s, flags=re.M)  # horizontal rules
    s = s.replace("**", "")                               # bold
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]{1,300}?)(?<!\s)\*(?![\w*])", r"\1", s)  # italics
    s = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", s, flags=re.M)  # bullets
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", s)  # links
    s = s.replace("`", "")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def verbatim(s):
    """Prompts pass through untouched except tab/CRLF safety."""
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")


def main(src, dst, copy=False):
    rows = json.load(open(src, encoding="utf-8"))
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quotechar='"',
                       quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(COLS)
        for i, r in enumerate(rows, start=1):
            w.writerow([
                r.get("id", i),
                verbatim(r.get("prompt", "")),
                to_plain(r.get("response", r.get("model_response", ""))).replace("\t", " "),
                r.get("model", r.get("model_name", "")),
                r.get("context", ""), r.get("interaction", ""), r.get("framing", ""),
                r.get("taxon", ""), r.get("salience", ""), r.get("cue", ""),
                r.get("score", ""),
            ])
    print(f"wrote {len(rows)} rows to {dst}")
    if copy:
        subprocess.run(["pbcopy"], input=open(dst, "rb").read(), check=True)
        print("ON YOUR CLIPBOARD — click a cell in Sheets and Cmd+V")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--copy"]
    if not args:
        sys.exit(__doc__.strip())
    src = args[0]
    dst = args[1] if len(args) > 1 else re.sub(r"\.json$", "", src) + ".sheet.tsv"
    main(src, dst, copy="--copy" in sys.argv)
