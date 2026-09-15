"""
Render FAILURE_MODES.md from the failure-mode registry.

    python render_failure_modes_md.py

The registry (scenario_schema.FAILURE_MODES) and the exemplar snapshot
(failure_modes/exemplars.tsv) are the sources; the markdown is a view. Re-run
after editing either.
"""

import os

from scenario_schema import FAILURE_MODES, FAILURE_MODE_TARGET, load_mode_exemplars

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FAILURE_MODES.md")


def render() -> str:
    ex = load_mode_exemplars()
    L: list[str] = []
    L.append("# WAP Failure Modes\n")
    L.append("Rendered from the registry in `scenario_schema.py` (`FAILURE_MODES`). Do not edit by hand; "
             "edit the registry and re-render:\n\n```\npython render_failure_modes_md.py\n```\n")
    L.append("Each generated scenario is assigned exactly one **failure mode** and one **variant** of it. The mode is the "
             "scenario's design goal: the text is engineered so that committing that particular failure is the natural careless "
             "response. The registry is transcribed from the team's failure-modes document (\"Allen - Failure Modes (animal welfare)\") "
             "and the example sheet (`failure_modes_prompt_examples`, snapshotted in `failure_modes/exemplars.tsv`). "
             "Those two are the source of truth; this file and the code follow them.\n")
    L.append("**Status.** Five modes are active. `instrumental_framing` and `harm_consideration` are under discussion and not "
             "yet in the registry; `reassurance` is out. The earlier CMEP-grader-derived taxonomy (14 modes including an "
             "over-consideration arm) was retired on 2026-09-08 in favour of this empirically grounded set.\n")
    L.append("**Mix.** Even across active modes: " + ", ".join(f"`{k}` {v:.0%}" for k, v in FAILURE_MODE_TARGET.items()) +
             ". Within a mode, variants are apportioned evenly; within a variant, contexts follow `CONTEXT_TARGET`.\n")
    L.append("---\n")
    for i, m in enumerate(FAILURE_MODES.values(), 1):
        L.append(f"## {i}. `{m.name}`\n")
        L.append(m.description[0].upper() + m.description[1:] + "\n")
        L.append("**Structure the text must contain.**")
        L += [f"- {el}" for el in m.structure]
        L.append("")
        L.append(f"**Mechanism of failure.** {m.mechanism}\n")
        L.append("**Commitments plausibly violated.**")
        L += [f"- {c}" for c in m.commitments]
        L.append("")
        L.append("**Variants.**")
        L += [f"- `{k}` — {v}" for k, v in m.variants.items()]
        L.append("")
        if m.siblings:
            L.append("**Siblings (distinguishing test).**")
            L += [f"- `{k}` — {v}" for k, v in m.siblings.items()]
            L.append("")
        cons = []
        if m.taxa:
            cons.append("taxa: " + ", ".join(m.taxa))
        if m.salience:
            cons.append("salience: " + ", ".join(f"{k} {v:.0%}" for k, v in m.salience.items()))
        cons.append("warranted: " + ", ".join(f"{k} {v:.0%}" for k, v in m.warranted.items()))
        if m.secondary_taxon:
            cons.append("two-species mode (cell carries `secondary_taxon_group`)")
        if m.text_check:
            cons.append(f"mechanical text check: `{m.text_check.__name__}`")
        L.append("**Cell constraints.** " + "; ".join(cons) + ".\n")
        picks = [e for e in ex.get(m.name, []) if e.get("philosophy_pick")]
        if picks:
            L.append("**Philosophy-team sample (from the sheet).**")
            for e in picks:
                hs = e.get("human_scores") or {}
                sc = f" (human score {min(hs.values()):g}–{max(hs.values()):g})" if hs else ""
                q = e["question"]
                L.append(f"- *{q[:220]}{'…' if len(q) > 220 else ''}* — `{e.get('variant') or '-'}`{sc}")
            L.append("")
        L.append("---\n")
    L.append("## Commitment → mode rollup\n")
    roll: dict[str, list[str]] = {}
    for m in FAILURE_MODES.values():
        for c in m.commitments:
            roll.setdefault(c.split(" — ")[0].strip(), []).append(m.name)
    L.append("| Commitment | Modes |\n|---|---|")
    for c, ms in sorted(roll.items()):
        L.append(f"| {c} | {', '.join(f'`{x}`' for x in ms)} |")
    L.append("")
    L.append("## Adding a mode\n")
    L.append("1. Append a `FailureMode(...)` to `FAILURE_MODES` in `scenario_schema.py`: description, structure, mechanism, "
             "commitments, variants, siblings, cell constraints, and (only if a lexical rule is reliable) a `text_check`.\n"
             "2. Add its tab to `GID_BY_MODE`, its philosophy picks and variant assignments in `sync_failure_mode_examples.py`, "
             "and re-run the sync to refresh `failure_modes/exemplars.tsv`.\n"
             "3. Add one scoring example for it in `scenario_prompts.SCORING_EXAMPLES`.\n"
             "4. `python scenario_generation.py --verify --verify-n 50 --probe` and read the per-mode block.\n"
             "5. `python render_failure_modes_md.py`.\n")
    return "\n".join(L)


if __name__ == "__main__":
    md = render()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT} ({len(md.split())} words)")
