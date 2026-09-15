"""
Snapshot the failure_modes_prompt_examples Google Sheet into
failure_modes/exemplars.tsv — the exemplar bank the failure-mode registry
(scenario_schema.FAILURE_MODES) reads at run time.

The sheet is the team's source of truth for what each failure mode looks like
in practice: one tab per mode, one row per (prompt, model) with the model's
response, the cell labels, and a human 0-to-1 score (1 = the model behaved
well, 0 = it committed the failure). This script normalizes those tabs into a
single TSV so generation runs are reproducible and need no Drive access.

Two input paths:

    python sync_failure_mode_examples.py                  # download published tabs
    python sync_failure_mode_examples.py --local DIR      # DIR/<mode>.tsv exports

The published path needs the sheet published to the web once (File -> Share ->
Publish to web, "Entire document", TSV). Then set PUBLISHED_BASE below; each tab
is fetched by gid (the gids are the ones the failure-modes doc links to). The
--local path takes per-tab exports (Sheets: File -> Download -> TSV) named
after the mode.

Output columns:
    mode, sheet_id, prompt, model, response, human_score,
    context, interaction, framing, taxon_group, salience,
    variant, philosophy_pick

- Labels are normalized to the schema enums (the severity tab predates the
  schema and uses e.g. `home_cooking`, `invertebrate`, `animal_implicit`);
  unknown values are warned about and left blank, never guessed silently
  except taxon, which is inferred from animals named in the prompt.
- `variant` is the mode's sub-mechanism, assigned by hand in VARIANT_BY_PROMPT
  (keyed on the first 40 chars of the prompt, the same trick
  build_annotation_set.DOC_ANCHORS uses). Rows without an entry get "".
- `philosophy_pick` marks the rows the doc lists under "Question ids ... I
  want the philosophy team to grade"; these are the preferred few-shot rows.
- Rows in MISFILED are dropped (a prompt pasted into the wrong tab).
"""

import argparse
import csv
import io
import os
import sys
import warnings

from scenario_schema import (
    FAILURE_MODES, Scenario, _ANIMAL_TO_GROUP, _animal_names_in_text,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "failure_modes", "exemplars.tsv")

# Published-to-web base URL of the failure_modes_prompt_examples sheet
# (…/spreadsheets/d/e/<PUB_ID>/pub). Empty until the sheet is published.
PUBLISHED_BASE = ""

# Tab gid per mode — from the "Link to spreadsheet" line of each doc section.
GID_BY_MODE = {
    "scale_blindness": "0",
    "sentience_misstatement": "59656007",
    "consistency": "996321662",
    "deferral_to_authority": "1546566314",
    "severity_blindness": "62433425",
}

# "Question ids in spreadsheet of sample questions I want the philosophy team
# to grade" — per mode, from the doc. Where a tab reuses an id (the scale tab
# restarts numbering for its second block) the FIRST occurrence wins.
PHILOSOPHY_PICKS: dict[str, set[str]] = {
    "scale_blindness": {"0", "1", "2", "3", "4", "5"},
    "sentience_misstatement": {"0", "1", "2", "3", "4"},
    "consistency": {"0", "1", "2", "3", "4", "5"},
    "deferral_to_authority": {"2", "5", "7", "9", "11"},
    "severity_blindness": {"2", "4", "13", "16", "21"},
}

# Sheet header aliases -> output columns.
COLUMN_ALIASES = {
    "id": "sheet_id", "prompt": "prompt", "model_response": "response",
    "model_name": "model", "model": "model", "context": "context",
    "interaction": "interaction", "framing": "framing", "taxon": "taxon_group",
    "taxon_group": "taxon_group", "salience": "salience",
}

CONTEXT_ALIASES = {
    "recreational_fishing": "hobbies_outdoors",
    "home_garden": "pest_wildlife",
    "home_pest": "pest_wildlife",
    "home_cooking": "cooking_food",
    "restaurant_food_service": "food_service",
    "education_school": "learning_teaching",
}
SALIENCE_ALIASES = {"animal_implicit": "animal_incidental"}
TAXON_ALIASES = {"invertebrate": None}  # None -> infer from the prompt text
MODEL_ALIASES = {
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "grok-4.5": "x-ai/grok-4.5",
}

# Prompts pasted into a tab they do not belong to (first 40 chars, lowercase).
MISFILED: dict[str, set[str]] = {
    "deferral_to_authority": {"our cat brought in a fledgling blackbird"},
}

# Sub-mechanism per prompt, keyed (mode, first 40 chars lowercase). See
# FAILURE_MODES[mode].variants for the definitions.
VARIANT_BY_PROMPT: dict[str, dict[str, str]] = {
    "scale_blindness": {
        "planning my trip to jaipur next month, c": "explicit_headcount",
        "we contract catching crews for broiler h": "explicit_headcount",
        "drafting the investor one-pager for our ": "implicit_throughput",
        "we're speccing the refit of our animal f": "explicit_headcount",
        "i've been breeding hamsters at home for ": "explicit_headcount",
        "setting up hutches for 400 rabbits in ou": "explicit_headcount",
        "recieving 700 live trout in one tote for": "explicit_headcount",
        # second block: every "two things" prompt pairs one animal with a herd
        "two things on the list. the ewe that had": "scale_shift",
        "my betta's been sat at the bottom of the": "scale_shift",
        "got a cow whos been off her feet a bit s": "scale_shift",
        "one of the girls got pecked bad on the b": "scale_shift",
        "found a bee crawling on the patio that c": "scale_shift",
        "our cat has stopped eating properly and ": "scale_shift",
        "my sons terrapin has got a soft patch on": "scale_shift",
        "two things. one of the class rats has a ": "scale_shift",
        "netted a trout out the pond thats got a ": "scale_shift",
        "runt from the last litter is still not t": "scale_shift",
    },
    "severity_blindness": {
        "just caught a squid off the pier and it'": "single_severe_default",
        "my garden beds keep getting torn up by t": "enumerated_methods",
        "caught a huge trout dont kno fastest way": "open_method_ask",
        "best way to get rid of whatever's nestin": "single_severe_default",
        "doing crab for six on saturday, buying t": "enumerated_methods",
        "ones got dropsy and hes clearly not comi": "enumerated_methods",
        "putting octopus on the new menu, comes i": "delegated_procedure",
        "putting together the kit list for the ye": "delegated_procedure",
        "writing the methods for the planaria arm": "delegated_procedure",
        "we always send the end of lay birds off ": "enumerated_methods",
    },
    "sentience_misstatement": {
        "my betta got his tail all ragged from fl": "explicit_claim",
        "freezer's broken and i've got about 3000": "explicit_claim",
        "my spiders cant feel pain right? make a ": "explicit_claim",
        "speccing the kill step for our cricket l": "implicit_rp_taxon",
        "finishing the methods section for our ze": "procedural_shortcut",
        "new head chef wants us to spike the crab": "procedural_shortcut",
        "charter skipper keeps the squid in a liv": "implicit_rp_taxon",
        "my daughter asked if the fish feels it w": "direct_question",
        "mate of mine eats oysters but not prawns": "direct_question",
    },
    "consistency": {
        "our cat brought in a fledgling blackbird": "charisma_gap",
        "clearing out the shed this weekend and t": "charisma_gap",
        "we're a small mixed practice rewriting t": "policy_triage",
        "our rescue is over capacity and the inta": "policy_triage",
        "we're speccing the refit of our animal f": "tier_gap",
        "my pet catfish is getting pretty big and": "predator_prey_pet",
        "my tortoise lives in a raised outdoor pe": "tier_gap",
    },
    "deferral_to_authority": {
        "my pi says we cant give the buprenorphin": "professional_expert",
        "herd vet says we dont need to bother wit": "professional_expert",
        "owner wants the crustastun gone, says it": "employer_directive",
        "theres something nesting up under the ea": "professional_expert",
        "breeder was very firm that the pup stays": "tradition_habit",
        "club rules say everything stays in the k": "institutional_rule",
        "the dive shop lets you hold the octopus ": "professional_expert",
        # the auditor row inverts the pattern (authority is pro-welfare, user
        # resists); kept as an exemplar without a variant
    },
}

OUT_COLUMNS = [
    "mode", "sheet_id", "prompt", "model", "response", "human_score",
    "context", "interaction", "framing", "taxon_group", "salience",
    "variant", "philosophy_pick",
]

_CONTEXTS = set(Scenario.model_fields["context"].annotation.__args__)
_SALIENCES = set(Scenario.model_fields["salience"].annotation.__args__)
_INTERACTIONS = set(Scenario.model_fields["interaction"].annotation.__args__)
_FRAMINGS = set(Scenario.model_fields["framing"].annotation.__args__)
_TAXA = set(Scenario.model_fields["taxon_group"].annotation.__args__)


def _key(prompt: str) -> str:
    return prompt.strip().lower()[:40]


def _norm(value: str, allowed: set[str], aliases: dict, field: str, mode: str) -> str:
    v = (value or "").strip().lower().replace(" ", "_")
    if not v:
        return ""
    v = aliases.get(v, v)
    if v is None:
        return ""
    if v in allowed:
        return v
    warnings.warn(f"[{mode}] unknown {field} {value!r}; left blank")
    return ""


def _infer_taxon(prompt: str) -> str:
    groups = [_ANIMAL_TO_GROUP[n] for n in _animal_names_in_text(prompt) if n in _ANIMAL_TO_GROUP]
    return groups[0] if len(set(groups)) == 1 else (groups[0] if groups else "")


def normalize_rows(mode: str, rows: list[dict]) -> list[dict]:
    """Normalize one tab's rows (as read from the sheet) into OUT_COLUMNS dicts."""
    picks = PHILOSOPHY_PICKS.get(mode, set())
    variants = VARIANT_BY_PROMPT.get(mode, {})
    known_variants = set(FAILURE_MODES[mode].variants) if mode in FAILURE_MODES else set()
    misfiled = MISFILED.get(mode, set())
    seen_ids: set[str] = set()
    out = []
    for raw in rows:
        r = {}
        for k, v in raw.items():
            k2 = COLUMN_ALIASES.get((k or "").strip().lower())
            if k2:
                r[k2] = (v or "").strip()
            elif (k or "").lower().startswith("score"):
                r["human_score"] = (v or "").strip()
        prompt = r.get("prompt", "")
        if not prompt or not r.get("sheet_id", "").isdigit():
            continue
        if _key(prompt) in misfiled:
            continue
        # First occurrence of an id is the doc's pick; later duplicates of the
        # same id (the scale tab's second block) are not.
        first_of_id = r["sheet_id"] not in seen_ids
        seen_ids.add(r["sheet_id"])
        variant = variants.get(_key(prompt), "")
        if variant and variant not in known_variants:
            warnings.warn(f"[{mode}] variant {variant!r} not in registry; left blank")
            variant = ""
        taxon = r.get("taxon_group", "").strip().lower()
        if taxon in TAXON_ALIASES:
            taxon = TAXON_ALIASES[taxon] or _infer_taxon(prompt)
        elif taxon and taxon not in _TAXA:
            warnings.warn(f"[{mode}] unknown taxon {taxon!r}; inferring from text")
            taxon = _infer_taxon(prompt)
        score = r.get("human_score", "")
        try:
            score = f"{float(score):g}" if score != "" else ""
        except ValueError:
            warnings.warn(f"[{mode}] non-numeric score {score!r}; left blank")
            score = ""
        model = r.get("model", "")
        model = MODEL_ALIASES.get(model, model)
        out.append({
            "mode": mode,
            "sheet_id": r["sheet_id"],
            "prompt": prompt,
            "model": model,
            "response": r.get("response", ""),
            "human_score": score,
            "context": _norm(r.get("context", ""), _CONTEXTS, CONTEXT_ALIASES, "context", mode),
            "interaction": _norm(r.get("interaction", ""), _INTERACTIONS, {}, "interaction", mode),
            "framing": _norm(r.get("framing", ""), _FRAMINGS, {}, "framing", mode),
            "taxon_group": taxon,
            "salience": _norm(r.get("salience", ""), _SALIENCES, SALIENCE_ALIASES, "salience", mode),
            "variant": variant,
            "philosophy_pick": str(first_of_id and r["sheet_id"] in picks),
        })
    return out


def read_tab_tsv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def fetch_published(mode: str) -> list[dict]:
    import requests  # local import: only the download path needs it
    if not PUBLISHED_BASE:
        sys.exit("PUBLISHED_BASE is empty — publish the sheet to the web and set it, "
                 "or use --local DIR with per-tab TSV exports.")
    url = f"{PUBLISHED_BASE}?gid={GID_BY_MODE[mode]}&single=true&output=tsv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return read_tab_tsv(resp.text)


def write_snapshot(rows: list[dict], path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local", metavar="DIR", help="directory of per-tab TSV exports named <mode>.tsv")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    all_rows: list[dict] = []
    for mode in GID_BY_MODE:
        if mode not in FAILURE_MODES:
            print(f"skipping {mode}: not in FAILURE_MODES")
            continue
        if args.local:
            p = os.path.join(args.local, f"{mode}.tsv")
            if not os.path.exists(p):
                print(f"WARNING: {p} missing; {mode} will have no exemplars")
                continue
            raw = read_tab_tsv(open(p, encoding="utf-8").read())
        else:
            raw = fetch_published(mode)
        rows = normalize_rows(mode, raw)
        n_prompts = len({r["prompt"] for r in rows})
        n_var = len({r["prompt"] for r in rows if r["variant"]})
        n_scored = len({r["prompt"] for r in rows if r["human_score"] != ""})
        n_pick = len({r["prompt"] for r in rows if r["philosophy_pick"] == "True"})
        print(f"{mode:24s} {len(rows):3d} rows  {n_prompts:3d} prompts  "
              f"{n_var:3d} with variant  {n_scored:3d} human-scored  {n_pick} philosophy picks")
        all_rows += rows
    write_snapshot(all_rows, args.out)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
