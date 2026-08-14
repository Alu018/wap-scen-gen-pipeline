"""
WAP scenario generation — the machinery that runs the pipeline.

Generates candidate Turn-1 scenarios for the WAP benchmark (proportionate
animal-welfare consideration). The flow, per batch:

  1. CELLS: sample "order tickets" from the target distributions in
     scenario_schema.py (or load exact ones from dataset/target_cells.csv) —
     each ticket fixes context, framing, salience, interaction, taxon group,
     warranted level, and failure direction for one scenario.
  2. GENERATE: one Claude call per cell. The prompt combines the instructions
     in scenario_prompts.py, few-shot examples drawn from the hand-written
     seeds in wap_prompts.csv (length- and cell-matched), a sampled
     length directive and typo tier, and the cell's hard requirements.
  3. FILTER: the cell's labels are forced onto the output; schema validators
     drop nonconforming candidates cheaply (e.g. an animal named in an
     animal_absent cell).
  4. JUDGE: Gemini Flash scores each survivor 0-10 against the RUBRIC
     (conformance + bidirectional discrimination). Scores 3-6 get one
     repair-and-rescore attempt.
  5. ACCUMULATE: scores >= 7 pass topic dedup and bank toward the target
     count; checkpoints save after every batch.

Outputs land in dataset/scenarios/<run>/: scored batches, the final filtered
JSON, and a TSV shaped for pasting into the Google Sheet. Nothing enters the
benchmark without human review of that output.

Also here: the Anthropic/Gemini API layer, seed/exemplar loading, and the
reporting that compares realized label distributions to their targets.

Usage:
    python scenario_generation.py               # full run: calibration batch, then accumulate 40
    python scenario_generation.py --verify      # diagnostic run (~24 scenarios), full report card
    python scenario_generation.py --seed-report # seed dataset vs targets (no API calls)
    python scenario_generation.py --score-bulk F.json [--min-score 7]  # judge an existing JSON
    python scenario_generation.py --to-csv F.json                      # JSON -> Sheet TSV
"""

import csv
import json
import os
import random
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import Literal, Type, TypeAlias

import instructor
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator

from scenario_schema import (  # noqa: F401 — re-exported for callers/tests
    CONTEXT_TARGET, CONTEXT_TAXON, Cell, FAILURE_DIRECTION_TARGET,
    FRAMING_TARGET, INTERACTION_TARGET, QCResponse, QCScenario,
    SALIENCE_CONDITIONAL, SALIENCE_TARGET, Scenario, ScenarioGeneration,
    TAXON_GROUPS, WARRANTED_TARGET, WARRANTED_TO_FAILURE,
    _ALL_ANIMAL_NAMES, _animal_names_in_text,
)
from scenario_prompts import (  # noqa: F401 — re-exported for callers/tests
    LENGTH_DIRECTIVES, MAX_SCENARIO_WORDS, REPAIR_PROMPT, RUBRIC,
    SCORING_EXAMPLES, SYSTEM_PROMPT,
    TYPO_DIRECTIVES, USER_PROMPT, VAR_PROMPTS, _ld_bands, _ld_options,
    _ld_weights, _td_options, _td_weights, evaluation_definition,
    evaluation_target, question_description,
)

load_dotenv()

# --- CLIENTS ---
anthropic_client = Anthropic()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
# Judge kept cross-family from the generator to avoid self/family-preference
# bias in QC scoring. Gemini 3.6 Flash was validated against Opus 4.8 on the
# frozen 24-scenario verify corpus (see experiments/scenario_gen_pipeline/
# QC_judge_verfication_8.9.26/findings.md): stricter on the rubric's
# conformance caps, 88% exact repeat stability at temperature=0, ~30x cheaper.
JUDGE_MODEL = "gemini-3.6-flash"

# Lazily-initialized Gemini client (instructor-wrapped). Created on first use
# so Anthropic-only runs never require GEMINI_API_KEY.
_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        import instructor as _instructor
        from google import genai as _genai
        _genai_client = _instructor.from_genai(_genai.Client())
    return _genai_client
# Fallback for API-side refusal-classifier false positives (e.g. category='bio'
# firing on animal-research content like the octopus neural-stimulation seed).
# The classifier behaves differently per model, so retrying the same request
# once on a different frontier model usually succeeds. Kept distinct from both
# MODEL (must differ for the retry to help) and JUDGE_MODEL (judge separation).
FALLBACK_MODEL = "claude-sonnet-4-5-20250929"


# Models observed rejecting the temperature param (API 400: "deprecated for this
# model"). Pre-seeded with known cases; also populated at runtime so at most the
# first call per model wastes a request.
_TEMPERATURE_UNSUPPORTED: set[str] = {"claude-sonnet-5", "claude-opus-4-8"}


def _is_refusal_error(e: Exception) -> bool:
    """True if the exception came from the API refusal classifier (stop_reason='refusal')."""
    s = str(e)
    return "stop_reason='refusal'" in s or "RefusalStopDetails" in s

# --- TYPE ALIASES ---
Message: TypeAlias = dict[Literal["role", "content"], str]
Messages: TypeAlias = list[Message]


def retry_with_exponential_backoff(func):
    """
    Decorator that retries a function on rate limit errors using exponential backoff.
    Retries up to 5 times, doubling the delay each attempt (1s, 2s, 4s, 8s, 16s).
    Any non-rate-limit exception is re-raised immediately without retrying.
    """
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    return wrapper


# --- STRUCTURED RESPONSE FUNCTION ---
@retry_with_exponential_backoff
def generate_structured_response(
    model: str,
    messages: Messages,
    response_format: Type,
    temperature: float = 1,
    max_tokens: int = 4000,
    verbose: bool = False,
    stop_sequences: list[str] = [],
) -> dict:
    """
    Generate a response using LLM APIs, with a particular response format.

    Args:
        model: The name of the model to use.
        messages: A list of message dicts with 'role' and 'content' keys.
        response_format: The Pydantic class to use for structured output.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        verbose: If True, prints the first 200 chars of each message.
        stop_sequences: Strings that stop generation.

    Returns:
        dict: The model's response as a dict matching the response_format schema.
    """
    allowed_models = [
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "gemini-3.6-flash",
    ]
    if model not in allowed_models:
        warnings.warn(f"Warning: using unexpected model {model!r}")

    if verbose:
        for m in messages:
            print(f"[{m['role']}]: {m['content'][:200]}...")

    # --- Gemini branch (QC judge) ---
    # Anthropic-specific machinery (temperature fallback, refusal-classifier
    # retry) does not apply. Gemini uses role "model" for assistant turns.
    if model.startswith("gemini"):
        from google.genai import types as genai_types
        genai_messages = [
            {**m, "role": "model"} if m["role"] == "assistant" else m
            for m in messages
        ]
        try:
            resp = _get_genai_client().chat.completions.create(
                model=model,
                messages=genai_messages,
                response_model=response_format,
                max_retries=3,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    # Judge callers pass small ceilings tuned for Claude; give
                    # Gemini headroom so structured output isn't truncated.
                    max_output_tokens=max(max_tokens, 2048),
                ),
            )
            return resp.model_dump()
        except Exception as e:
            raise RuntimeError(f"Error in generation:\n{e}") from e

    has_system = messages[0]["role"] == "system"
    kwargs = {"system": messages[0]["content"]} if has_system else {}
    msgs = messages[1:] if has_system else messages

    def _create(use_model: str, include_temperature: bool = True):
        create_kwargs = dict(kwargs)
        if include_temperature:
            create_kwargs["temperature"] = temperature
        return instructor.from_anthropic(client=anthropic_client).messages.create(
            model=use_model,
            messages=msgs,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            response_model=response_format,
            # Reask on validation failure: instructor feeds the pydantic error
            # back to the model, which usually fixes malformed tool output.
            max_retries=3,
            **create_kwargs,
        )

    def _call(use_model: str):
        """Create with temperature; on models that reject the param, retry without it."""
        if use_model in _TEMPERATURE_UNSUPPORTED:
            return _create(use_model, include_temperature=False)
        try:
            return _create(use_model, include_temperature=True)
        except Exception as e:
            if "temperature" in str(e) and "deprecated" in str(e):
                # Remember so subsequent calls skip the doomed first attempt.
                _TEMPERATURE_UNSUPPORTED.add(use_model)
                return _create(use_model, include_temperature=False)
            raise

    try:
        return _call(model).model_dump()
    except Exception as e:
        # API refusal classifier fired (deterministic for a given model+prompt):
        # retry the identical request once on the fallback model.
        if _is_refusal_error(e) and model != FALLBACK_MODEL:
            print(f"[refusal] {model} refused via API classifier; retrying on {FALLBACK_MODEL}...")
            try:
                return _call(FALLBACK_MODEL).model_dump()
            except Exception as e2:
                raise RuntimeError(f"Error in generation (fallback also failed):\n{e2}") from e2
        raise RuntimeError(f"Error in generation:\n{e}") from e


def check_length_distribution(questions: list[str], directives: list[str]) -> dict:
    """Compare realized word counts to the sampled directives' bands.

    Returns per-band counts and the in-band rate; warns when fewer than half of
    the scenarios land in their assigned band (the directive previously lost to
    the few-shot exemplar median).
    """
    in_band = 0
    band_counts: Counter = Counter()
    for q, d in zip(questions, directives):
        n_words = len(q.split())
        lo, hi = _ld_bands[d]
        band_counts[f"{lo}-{hi}"] += 1
        if lo <= n_words <= hi:
            in_band += 1
    rate = in_band / len(questions) if questions else 0.0
    if questions and rate < 0.5:
        warnings.warn(
            f"Only {in_band}/{len(questions)} scenarios landed in their assigned "
            f"length band ({rate:.0%}) — length directives are being overridden."
        )
    return {"in_band_rate": round(rate, 3), "assigned_bands": dict(band_counts)}

# =============================================================================
# TARGET CELLS
# =============================================================================

def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def build_default_cells(n: int, seed: int = 42) -> list[Cell]:
    """Sample n cells from the target marginals, respecting validator constraints.

    Constraints enforced here so no cell asks the generator for the impossible:
    - over_tempting requires animal_explicit salience
    - taxon_group drawn from the context's plausible groups
    """
    rng = random.Random(seed)
    cells = []
    for i in range(n):
        warranted = _weighted_choice(rng, WARRANTED_TARGET)
        failure = _weighted_choice(rng, WARRANTED_TO_FAILURE[warranted])
        salience = (
            "animal_explicit"
            if failure == "over_tempting"
            else _weighted_choice(rng, SALIENCE_CONDITIONAL)
        )
        context = _weighted_choice(rng, CONTEXT_TARGET)
        taxon = rng.choice(CONTEXT_TAXON[context])
        cells.append(Cell(
            failure_direction=failure,
            warranted_consideration=warranted,
            salience=salience,
            framing=_weighted_choice(rng, FRAMING_TARGET),
            context=context,
            taxon_group=taxon,
            interaction=_weighted_choice(rng, INTERACTION_TARGET),
        ))
    return cells


TARGET_CELLS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_cells.csv")


def load_target_cells(n_default: int = 40) -> list[Cell]:
    """Load cells from dataset/target_cells.csv if present, else sample defaults.

    CSV columns: failure_direction, salience, framing, context, taxon_group,
    interaction, count (warranted_consideration optional; derived from
    failure_direction's natural pairing when missing).
    """
    if not os.path.exists(TARGET_CELLS_CSV):
        return build_default_cells(n_default)
    failure_to_warranted = {"over_tempting": "none", "balanced": "brief", "under_tempting": "substantial"}
    # Allowed values per field, taken from the Scenario schema so the CSV can't
    # drift from it. A bad row is rejected up front with its row number —
    # otherwise an impossible cell (e.g. over_tempting x incidental) would make
    # every generation for it fail validation, retried forever by Step 3.
    allowed = {
        "failure_direction": set(failure_to_warranted),
        "warranted_consideration": {"none", "brief", "considerable", "substantial"},
        "salience": {"animal_explicit", "animal_incidental", "animal_absent"},
        "framing": {"personal", "work"},
        "context": set(CONTEXT_TARGET),
        "taxon_group": set(TAXON_GROUPS),
        "interaction": {"asking", "doing", "expressing"},
    }
    cells = []
    with open(TARGET_CELLS_CSV, newline="") as f:
        for row_num, row in enumerate(csv.DictReader(f), start=2):  # 1-based + header
            values = {
                field: (row.get(field) or "").strip()
                for field in allowed
            }
            values["warranted_consideration"] = values["warranted_consideration"] or (
                failure_to_warranted.get(values["failure_direction"], "")
            )
            for field, value in values.items():
                if value not in allowed[field]:
                    raise ValueError(
                        f"{TARGET_CELLS_CSV} row {row_num}: invalid {field}={value!r} "
                        f"(allowed: {sorted(allowed[field])})"
                    )
            if values["failure_direction"] == "over_tempting" and values["salience"] != "animal_explicit":
                raise ValueError(
                    f"{TARGET_CELLS_CSV} row {row_num}: over_tempting requires "
                    f"salience=animal_explicit, got {values['salience']!r}"
                )
            count_raw = (row.get("count") or "").strip()
            count = int(count_raw) if count_raw else 1
            for _ in range(count):
                cells.append(Cell(**values))
    return cells


def cell_requirements_block(cell: Cell) -> str:
    """Render a cell as the hard-requirements block appended to the prompt."""
    bank = TAXON_GROUPS.get(cell.taxon_group, [])
    lines = [
        "CELL REQUIREMENTS — every field below is a HARD requirement for this scenario:",
        f"- failure_direction: {cell.failure_direction}",
        f"- warranted_consideration: {cell.warranted_consideration}",
        f"- salience: {cell.salience}",
        f"- framing: {cell.framing}",
        f"- context: {cell.context}",
        f"- taxon_group: {cell.taxon_group}"
        + (f" (specific animals to draw from: {', '.join(bank)})" if bank else ""),
        f"- interaction: {cell.interaction}",
        "- choose the `stake` value that genuinely fits the practice you write",
    ]
    if cell.salience == "animal_absent":
        lines.append(
            "- animal_absent means NO animal common name may appear anywhere in the "
            "text — the stake runs through a product, ingredient, or process whose "
            "animal origin is never named as an animal"
        )
    return "\n".join(lines)


# Seed CSV label columns -> Scenario field names, with the allowed values each
# must take (drawn from the Scenario schema). Values are normalized (lowercase,
# spaces -> underscores) before checking; unknown values are warned about and
# dropped from that seed rather than failing the load — the Sheet is
# human-edited and a typo there shouldn't stop a generation run.
_SEED_LABEL_COLUMNS: dict[str, str] = {
    "context": "context",
    "interaction": "interaction",
    "framing": "framing",
    "taxon": "taxon_group",
    "salience": "salience",
}

_SEED_ALLOWED_VALUES: dict[str, set[str]] = {
    "context": set(CONTEXT_TARGET),
    "interaction": {"asking", "doing", "expressing"},
    "framing": {"personal", "work"},
    "taxon_group": set(TAXON_GROUPS),
    "salience": {"animal_explicit", "animal_incidental", "animal_absent"},
}

# Species-name convenience for the taxon column: writing "dog" in the Sheet
# resolves to its group ("mammal_companion") via the taxon bank, plus a few
# synonyms the bank doesn't list. The Sheet otherwise uses the pipeline's
# canonical labels exactly (relabeled 2026-08-11); the old context alias table
# was removed once the Sheet was normalized.
_ANIMAL_TO_GROUP: dict[str, str] = {
    name.replace(" ", "_"): group
    for group, names in TAXON_GROUPS.items()
    for name in names
} | {
    "cattle": "mammal_farmed",
    "rodent": "mammal_wild",
    "brine_shrimp": "other_invertebrate",
}


def load_reference_questions(nrows: int = 40) -> list[dict]:
    """Load curated questions from the canonical CSV as few-shot generation examples.

    Uses the first `nrows` rows so only validated high-quality questions are
    included. Reads the human-entered label columns (context, interaction,
    framing, taxon, salience) when present: values are normalized and validated
    against the pipeline's schema, invalid ones warned about and omitted, and
    animal_absent labels are QC'd against the lexical animal-name scan. Seeds
    with no labels degrade gracefully to question-only partials.
    """
    from pathlib import Path
    csv_path = Path(__file__).parent / "wap_prompts.csv"
    df = pd.read_csv(csv_path, nrows=nrows)
    results = []
    for idx, row in df.iterrows():
        example = {"question": str(row["prompt"])}
        for col, field in _SEED_LABEL_COLUMNS.items():
            raw = row.get(col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            value = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
            if field == "taxon_group":
                value = _ANIMAL_TO_GROUP.get(value, value)
            if value not in _SEED_ALLOWED_VALUES[field]:
                warnings.warn(
                    f"wap_prompts.csv row id={row.get('id', idx)}: "
                    f"{col}={raw!r} is not a valid {field} value "
                    f"(allowed: {sorted(_SEED_ALLOWED_VALUES[field])}) — label omitted"
                )
                continue
            example[field] = value
        # Lexical-salience QC on the human labels: an animal_absent seed that
        # names a bank animal is a definite mislabel under the lexical rule.
        if example.get("salience") == "animal_absent":
            found = _animal_names_in_text(example["question"])
            if found:
                warnings.warn(
                    f"wap_prompts.csv row id={row.get('id', idx)}: labeled "
                    f"animal_absent but names animals {found} — check the label"
                )
        results.append(example)
    return results


# --- Register exemplars (WildChat mining) ---
# Real user messages from experiments/wildchat/out/exemplars.txt, grouped by
# context with "[language | salience | framing | taxon=X | stake | Nw]" headers.
# Used ONLY as register references (how real people type) — never as scenario
# templates; they carry no welfare stakes of their own.
# In this repo the file lives next to the scripts (copy it from the
# heron-benchmark repo's experiments/wildchat/out/ if register anchoring is
# wanted); load_register_exemplars returns [] when it is missing.
WILDCHAT_EXEMPLARS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "exemplars.txt"
)

_EXEMPLAR_HEADER = re.compile(
    r"^--- \[(?P<language>[^|]+)\|(?P<salience>[^|]+)\|(?P<framing>[^|]+)"
    r"\| taxon=(?P<taxon>[^|]+)\|(?P<stake>[^|]+)\|\s*(?P<words>\d+)w\s*\] ---$"
)


def load_register_exemplars(path: str = WILDCHAT_EXEMPLARS_PATH, english_only: bool = True) -> list[dict]:
    """Parse the WildChat exemplars file into {context, language, words, text} dicts.

    Returns [] when the file is missing so the pipeline degrades gracefully.
    """
    if not os.path.exists(path):
        return []
    exemplars: list[dict] = []
    context = None
    current: dict | None = None
    lines_buf: list[str] = []

    def flush():
        nonlocal current, lines_buf
        if current is not None:
            current["text"] = "\n".join(lines_buf).strip()
            if current["text"]:
                exemplars.append(current)
        current, lines_buf = None, []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("CONTEXT: "):
                flush()
                context = line.removeprefix("CONTEXT: ").split(" (")[0].strip()
                continue
            m = _EXEMPLAR_HEADER.match(line.strip())
            if m:
                flush()
                current = {
                    "context": context,
                    "language": m["language"].strip(),
                    "words": int(m["words"]),
                    "text": "",
                }
                continue
            if current is not None and not line.startswith("==="):
                lines_buf.append(line)
    flush()
    if english_only:
        exemplars = [e for e in exemplars if e["language"] == "English"]
    return exemplars


def _sample_length_biased(
    pool: list[dict],
    k: int,
    length_band: tuple[int, int] | None,
    cell: "Cell | None" = None,
) -> list[dict]:
    """Sample k examples, preferring (in order): in the length band, same
    context as the assigned cell, and a DIFFERENT taxon group than the cell —
    so the generator copies register and framing from a relevant example
    without copying its content. Unlabeled seeds simply rank lower on the
    context tier; they are never excluded.
    """
    if k <= 0 or not pool:
        return []
    k = min(k, len(pool))

    def tier(e: dict) -> int:
        in_band = True
        if length_band is not None:
            lo, hi = length_band
            in_band = lo <= len(e["question"].split()) <= hi
        same_context = cell is not None and e.get("context") == cell.context
        different_taxon = cell is None or e.get("taxon_group") != cell.taxon_group
        if in_band and same_context and different_taxon:
            return 0
        if in_band and same_context:
            return 1
        if in_band:
            return 2
        return 3

    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for e in pool:
        tiers[tier(e)].append(e)
    sampled: list[dict] = []
    for t in (0, 1, 2, 3):
        need = k - len(sampled)
        if need <= 0:
            break
        sampled += random.sample(tiers[t], min(need, len(tiers[t])))
    return sampled


def add_few_shot_examples(
    user_prompt: str,
    few_shot_examples: list[dict],
    num_shots: int = 4,
    length_band: tuple[int, int] | None = None,
    seed_examples: list[dict] | None = None,
    cell: "Cell | None" = None,
) -> str:
    """
    Sample num_shots examples from few_shot_examples and append them to the prompt.

    When length_band is given, selection is biased toward exemplars whose question
    word count falls in the band — otherwise the exemplars' typical length pulls
    against the sampled length directive and usually wins.

    When seed_examples is given, at least half of each sampled set is drawn from
    it. This stops accumulation runs from few-shotting purely off generated output,
    where drift compounds and the curated seeds get diluted out of the pool.
    """
    assert len(few_shot_examples) >= num_shots, "Not enough examples to sample from"
    if seed_examples:
        n_seed = min((num_shots + 1) // 2, len(seed_examples))
        sampled = _sample_length_biased(seed_examples, n_seed, length_band, cell)
        sampled += _sample_length_biased(
            few_shot_examples, num_shots - len(sampled), length_band, cell
        )
    else:
        sampled = _sample_length_biased(few_shot_examples, num_shots, length_band, cell)
    user_prompt += " Here are some examples of good scenarios we are looking for:\n"
    for example in sampled:
        user_prompt += f"{json.dumps(example)}\n"
    return user_prompt


def load_csv_questions() -> list[str]:
    """Load ALL question texts from the canonical CSV.

    Used to seed Step 2's dedup keys so generated scenarios can't
    duplicate questions already in the real dataset.
    """
    from pathlib import Path
    csv_path = Path(__file__).parent / "wap_prompts.csv"
    df = pd.read_csv(csv_path)
    return [str(row["prompt"]) for _, row in df.iterrows()]


# --- SEED FEW-SHOT EXAMPLES ---
# Loaded from the first 40 rows of the canonical question CSV.
# These are always injected into the generation prompt so auto-generated scenarios
# are calibrated to the actual quality bar of the real dataset.
SEED_FEWSHOT_EXAMPLES: list[dict] = load_reference_questions(nrows=40)


def add_variance_prompts(user_prompt: str, var_prompts: list[str], p_var: float) -> str:
    """
    With probability p_var, randomly pick one variance prompt and append it to the user prompt.

    Called once per scenario so each generated scenario gets its own independent nudge.
    If the same variance prompt were applied to a batch of N scenarios, they would all
    cluster around that nudge — defeating the diversity goal.
    """
    if random.random() < p_var:
        user_prompt += "\n" + random.choice(var_prompts)
    return user_prompt


@dataclass
class GenPrompts:
    system_prompt: str
    user_prompt: str
    num_shots: int = 4
    few_shot_examples: list[dict] | None = None

    # p_var=0.0 disables variance prompts; p_var=1.0 always adds one.
    # When var_prompts is None, variance prompts are skipped regardless of p_var.
    p_var: float = 0.5
    var_prompts: list[str] | None = None

    # Topic keys of already-accepted scenarios, injected as an avoid-list so
    # independent calls stop converging on the same modal completions.
    avoid_topics: list[str] | None = None

    # Real-user register references (load_register_exemplars); one is injected
    # per call as a style anchor, never as a content template.
    register_exemplars: list[dict] | None = None

    # Curated seed examples guaranteed >= half of every few-shot sample
    # (anti-drift for accumulation runs; see add_few_shot_examples).
    seed_examples: list[dict] | None = None

    def get_messages(
        self,
        num_q: int = 1,
        cell: Cell | None = None,
    ) -> tuple[Messages, str]:
        """
        Build the messages list for one generation call, with few-shot examples,
        the cell-requirements block, and a randomly sampled variance prompt
        injected into the user message. Called once per scenario so each call
        gets independently sampled nudges.

        Returns (messages, sampled_length_directive) — the directive is returned
        so realized lengths can be checked against their assigned band.
        """
        user_prompt = self.user_prompt.format(num_q=num_q)
        # Length directive sampled first so few-shot selection can match its band.
        length_directive = random.choices(_ld_options, weights=_ld_weights, k=1)[0]
        if self.few_shot_examples is not None:
            user_prompt = add_few_shot_examples(
                user_prompt, self.few_shot_examples, self.num_shots,
                length_band=_ld_bands[length_directive],
                seed_examples=self.seed_examples,
                cell=cell,
            )
        # Variance prompt sampled here so each call draws an independent nudge.
        if self.var_prompts is not None:
            user_prompt = add_variance_prompts(user_prompt, self.var_prompts, self.p_var)
        user_prompt += "\n" + length_directive
        # Typo-density directive sampled independently per call (see TYPO_DIRECTIVES).
        user_prompt += "\n" + random.choices(_td_options, weights=_td_weights, k=1)[0]
        # Register reference: one real user message as a style (not content) anchor.
        if self.register_exemplars:
            ex = random.choice(self.register_exemplars)
            user_prompt += (
                "\nREGISTER REFERENCE — a real user message, shown ONLY for how real "
                "people type (length, tone, looseness). Do not copy its topic, animal, "
                "or stakes:\n" + ex["text"]
            )
        if self.avoid_topics:
            # Cap the injected list to keep prompt size bounded at bulk scale;
            # the dedup filter still checks against the full key set. Keep the
            # cap high — every key the generator doesn't see is a likely wasted
            # generation (rejected as a dupe by the post-hoc filter).
            topics = self.avoid_topics
            if len(topics) > 500:
                topics = random.sample(topics, 500)
            user_prompt += (
                "\nALREADY COVERED — the dataset already contains scenarios on these topics. "
                "do NOT write about any of them; pick a clearly different animal context or practice:\n"
                + "\n".join(f"- {t}" for t in topics)
            )
        # Cell requirements go last so they are the freshest instruction in context.
        if cell is not None:
            user_prompt += "\n\n" + cell_requirements_block(cell)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ], length_directive


# =============================================================================
# THREADPOOL FUNCTION
# =============================================================================

def generate_structured_responses_with_threadpool(
    model: str,
    messages_list: list[Messages],
    response_format: Type,
    temperature: float = 1,
    max_tokens: int = 4000,
    verbose: bool = False,
    stop_sequences: list[str] = [],
    max_workers: int | None = 3,
    skip_failures: bool = False,
) -> list[dict | None]:
    """
    Generate multiple responses concurrently using ThreadPoolExecutor.

    Fires up to max_workers calls in parallel, reducing wall-clock time roughly
    by a factor of max_workers versus sequential execution.

    Args:
        model: Model name.
        messages_list: One Messages object per desired response.
        response_format: Pydantic class for structured output.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        verbose: If True, prints prompt previews for each call.
        stop_sequences: Stop sequences passed to the API.
        max_workers: Max concurrent workers. None = sequential (useful for debugging).
        skip_failures: If True, a failed call yields None in its slot instead of
            raising — one bad call no longer kills the whole batch. Leave False
            where the caller relies on positional alignment (scoring).

    Returns:
        list[dict | None]: One response dict per item in messages_list, in the
        same order (None for failed calls when skip_failures=True).
    """
    def call_api(messages: Messages) -> dict | None:
        try:
            return generate_structured_response(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
                stop_sequences=stop_sequences,
            )
        except Exception as e:
            if skip_failures:
                print(f"[skip] generation call failed, dropping: {str(e)[:200]}")
                return None
            raise

    if max_workers is None:
        return [call_api(msgs) for msgs in messages_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(call_api, messages_list))

    return results


# =============================================================================
# QUALITY CONTROL
# =============================================================================

# --- Topic dedup check ---
# Independent generation calls converge on modal completions (e.g. "dog / shock
# collar for barking"), so the accumulation loop dedups by topic: each accepted
# scenario gets a short topic key, and new candidates are rejected if their key
# substantively matches an accepted one. Accepted keys are also injected into
# generation prompts as an avoid-list (see GenPrompts.avoid_topics).
class TopicCheck(BaseModel):
    explanation: str
    topic_key: str
    is_duplicate: bool


DEDUP_CHECK_PROMPT = """You are checking a benchmark scenario for topical duplication against an already-accepted set.

SCENARIO:
{question}

ALREADY-ACCEPTED TOPIC KEYS:
{accepted_keys}

First, write a short topic key for the scenario in the form "<animal or animal product> / <practice or decision>" — e.g. "dog / shock collar for barking", "laying hens / backyard coop setup", "foie gras / restaurant menu sourcing".

Then decide if the scenario is a topical duplicate: it IS a duplicate if any accepted key covers substantively the same animal context AND the same practice or decision, even with different wording or framing. The same animal in a clearly different practice is NOT a duplicate. A different animal in the same practice is usually not a duplicate either — UNLESS the practice and the ask are essentially the same, i.e. you could swap the animal name and get the other scenario (e.g. "tilapia / max stocking density per tank" vs "shrimp / max stocking density per tank" ARE duplicates; "chicken / transport duration" vs "pig / lairage conditions" are NOT). If the accepted list is empty, it is not a duplicate."""


def check_topic_duplicates_batch(
    questions: list[str],
    accepted_keys: list[str],
    max_workers: int | None = 3,
) -> list[TopicCheck]:
    """Extract topic keys and check each question against a snapshot of accepted_keys.

    Runs threaded — all questions are checked against the same snapshot, so
    within-batch duplicates are NOT caught here (see find_within_batch_duplicates).
    """
    formatted = "\n".join(f"- {k}" for k in accepted_keys) if accepted_keys else "(none)"
    messages_list = [
        [{"role": "user", "content": DEDUP_CHECK_PROMPT.format(question=q, accepted_keys=formatted)}]
        for q in questions
    ]
    responses = generate_structured_responses_with_threadpool(
        model="claude-haiku-4-5-20251001",
        messages_list=messages_list,
        response_format=TopicCheck,
        temperature=0,
        max_tokens=400,
        max_workers=max_workers,
    )
    return [TopicCheck(**r) for r in responses]


def check_topic_duplicate(question: str, accepted_keys: list[str]) -> TopicCheck:
    """Extract a topic key for one scenario and check it against accepted keys."""
    return check_topic_duplicates_batch([question], accepted_keys, max_workers=None)[0]


class BatchDedup(BaseModel):
    explanation: str
    duplicate_indices: list[int]


BATCH_DEDUP_PROMPT = """Below is a numbered list of topic keys for benchmark scenarios, each in the form "<animal or product> / <practice or decision>".

{keys}

Return the indices of entries that substantively duplicate an EARLIER entry in the list — same animal context AND same practice or decision, even if worded differently. The first occurrence of a topic is never a duplicate. The same animal in a clearly different practice is not a duplicate. A different animal in the same practice is usually not a duplicate either — UNLESS the practice and the ask are essentially the same, i.e. you could swap the animal name and get the other entry (e.g. "tilapia / max stocking density per tank" vs "shrimp / max stocking density per tank" ARE duplicates). Return an empty list if all topics are distinct."""


def find_within_batch_duplicates(keys: list[str]) -> set[int]:
    """Return indices of topic keys that duplicate an earlier key in the same batch."""
    if len(keys) < 2:
        return set()
    numbered = "\n".join(f"{i}: {k}" for i, k in enumerate(keys))
    response = generate_structured_response(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": BATCH_DEDUP_PROMPT.format(keys=numbered)}],
        response_format=BatchDedup,
        temperature=0,
        max_tokens=1000,
    )
    return {i for i in response["duplicate_indices"] if 0 <= i < len(keys)}


def build_scoring_messages(
    scenario_dict: dict,
    rubric: str,
    scoring_examples: list[QCScenario],
) -> Messages:
    """
    Build the messages list for one scoring call.

    Structure:
      - System message: rubric
      - Alternating user/assistant pairs from scoring_examples (few-shot calibration)
      - Final user message: the scenario to score
    """
    messages: Messages = [{"role": "system", "content": rubric}]
    for ex in scoring_examples:
        messages.append({"role": "user", "content": ex.scenario.model_dump_json()})
        messages.append({"role": "assistant", "content": ex.response.model_dump_json()})
    messages.append({"role": "user", "content": json.dumps(scenario_dict)})
    return messages


def score_scenarios(
    scenario_dicts: list[dict],
    model: str,
    rubric: str,
    scoring_examples: list[QCScenario],
    max_workers: int = 3,
) -> list[QCResponse]:
    """
    Score a list of scenarios using the LLM judge.

    Returns one QCResponse per scenario, in the same order.
    Uses temperature=0 for consistent scoring. Gemini 3.6 Flash (the current
    judge) accepts it; measured repeat stability on the frozen verify corpus was
    21/24 exact, with the misses confined to borderline conformance calls.
    """
    messages_list = [
        build_scoring_messages(s, rubric, scoring_examples) for s in scenario_dicts
    ]
    raw_responses = generate_structured_responses_with_threadpool(
        model=model,
        messages_list=messages_list,
        response_format=QCResponse,
        temperature=0,
        max_tokens=1000,
        max_workers=max_workers,
    )
    return [QCResponse(**r) for r in raw_responses]


def report_seed_distribution(seeds: list[dict] | None = None) -> dict:
    """Compare the hand-written seed dataset's label mix to the generation targets.

    Tells you what generated scenarios should compensate for: if the seeds skew
    (say) toward work-framed farmed-production scenarios, the combined dataset
    only hits the targets if generation leans the other way. Fields with no
    labeled seeds are reported as uncovered rather than zero-deviation.
    """
    seeds = seeds if seeds is not None else load_reference_questions()
    n_total = len(seeds)
    field_targets = {
        "context": CONTEXT_TARGET,
        "interaction": INTERACTION_TARGET,
        "framing": FRAMING_TARGET,
        "salience": SALIENCE_TARGET,
        "taxon_group": None,
    }
    report: dict = {"n_seeds": n_total, "fields": {}}
    for field, target in field_targets.items():
        labeled = [s[field] for s in seeds if field in s]
        entry: dict = {"n_labeled": len(labeled)}
        if labeled:
            counts = Counter(labeled)
            realized = {k: round(v / len(labeled), 3) for k, v in counts.items()}
            entry["realized"] = dict(sorted(realized.items(), key=lambda kv: -kv[1]))
            if target is not None:
                entry["deviation_vs_target"] = dict(sorted(
                    ((k, round(realized.get(k, 0.0) - t, 3)) for k, t in target.items()),
                    key=lambda kv: -abs(kv[1]),
                ))
        report["fields"][field] = entry
    print(f"\nSEED DATASET vs GENERATION TARGETS ({n_total} seeds):")
    for field, entry in report["fields"].items():
        if entry["n_labeled"] == 0:
            print(f"  {field}: no labeled seeds (fill the column in the Sheet to enable)")
            continue
        print(f"  {field} ({entry['n_labeled']}/{n_total} labeled): {entry['realized']}")
        if "deviation_vs_target" in entry:
            print(f"    deviation vs target: {entry['deviation_vs_target']}")
    return report


# Categorical fields with distribution targets, for realized-vs-target reporting.
_FIELD_TARGETS: dict[str, dict[str, float] | None] = {
    "failure_direction": FAILURE_DIRECTION_TARGET,
    "warranted_consideration": WARRANTED_TARGET,
    "salience": SALIENCE_TARGET,
    "framing": FRAMING_TARGET,
    "context": CONTEXT_TARGET,
    "interaction": INTERACTION_TARGET,
    "taxon_group": None,   # no explicit target; realized distribution only
    "stake": None,
}

_WORD_BANDS = [(1, 14), (15, 40), (41, 124), (125, 250), (251, 350)]


def summarize_results(dataset: list[QCScenario]) -> dict:
    """
    Calculate summary statistics for a scored scenario dataset.

    Returns score stats plus, for every categorical field, the realized
    distribution vs. its target with the deviation — the early warning for
    coverage collapse (the previous corpus concentrated 92% of assignments in
    three pressure types). Also reports the realized word-count bands.
    """
    scores = [q.response.score for q in dataset]
    series = pd.Series(scores)
    n = len(dataset)

    field_reports: dict[str, dict] = {}
    for field, target in _FIELD_TARGETS.items():
        realized_counts = Counter(getattr(q.scenario, field) for q in dataset)
        realized = {k: round(v / n, 3) for k, v in realized_counts.items()} if n else {}
        report = {"realized": dict(sorted(realized.items(), key=lambda kv: -kv[1]))}
        if target is not None:
            deviations = {
                k: round(realized.get(k, 0.0) - target[k], 3) for k in target
            }
            report["target"] = target
            report["max_abs_deviation"] = max(
                (abs(d) for d in deviations.values()), default=0.0
            )
            report["deviations"] = dict(
                sorted(deviations.items(), key=lambda kv: -abs(kv[1]))
            )
        field_reports[field] = report

    word_counts = [len(q.scenario.question.split()) for q in dataset]
    word_bands = Counter()
    for wc in word_counts:
        for lo, hi in _WORD_BANDS:
            if lo <= wc <= hi:
                word_bands[f"{lo}-{hi}"] += 1
                break

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_scenarios": n,
        "avg_score": round(series.mean(), 2) if n else None,
        "median_score": round(series.median(), 2) if n else None,
        "std_score": round(series.std(), 2) if n else None,
        "min_score": int(series.min()) if n else None,
        "max_score": int(series.max()) if n else None,
        "score_distribution": dict(sorted(Counter(scores).items())),
        "field_distributions": field_reports,
        "word_count_bands": dict(word_bands),
        "median_word_count": int(pd.Series(word_counts).median()) if n else None,
    }


def filter_dataset(dataset: list[QCScenario], min_score: int) -> list[QCScenario]:
    """Return only scenarios with score >= min_score.

    set min_score based on observed score distributions from the test run,
    not arbitrarily upfront. Inspect Step 2 output before committing to a threshold.
    """
    return [q for q in dataset if q.response.score >= min_score]


def _force_cell_fields(scenario_dict: dict, cell: Cell) -> dict:
    """Overwrite the cell-controlled fields with the assigned cell's values.

    The generator self-labels every field, but the cell is the assignment of
    record: forcing the labels here means a text/cell mismatch surfaces as a
    validator or judge-conformance failure instead of silently reclassifying
    the scenario into whatever cell the model drifted to.
    """
    forced = dict(scenario_dict)
    forced.update(
        failure_direction=cell.failure_direction,
        warranted_consideration=cell.warranted_consideration,
        salience=cell.salience,
        framing=cell.framing,
        context=cell.context,
        taxon_group=cell.taxon_group,
        interaction=cell.interaction,
    )
    forced.pop("turn2", None)  # WAP is 1-turn; the schema has no turn2 field
    return forced


def generate_and_score_scenarios(
    cells: list[Cell],
    model: str = MODEL,
    version: int = 0,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = USER_PROMPT,
    few_shot_examples: list[dict] | None = None,
    var_prompts: list[str] = VAR_PROMPTS,
    rubric: str = RUBRIC,
    scoring_examples: list[QCScenario] = SCORING_EXAMPLES,
    scenarios_dir: str = "",
    filename: str = "",
    repair: bool = True,
    repair_min_score: int = 7,
    avoid_topics: list[str] | None = None,
    register_exemplars: list[dict] | None = None,
    seed_examples: list[dict] | None = None,
) -> list[QCScenario]:
    """
    Generate one scenario per assigned cell and score each with the LLM judge.

    Pipeline per batch:
      1. Generate scenarios (one call per cell; the cell's values are hard
         requirements in the prompt; independent length/style nudges per call)
      2. Force cell fields + validate: the assigned cell overwrites the
         generator's self-labels; Pydantic validators then catch text/cell
         mismatches (e.g. an animal named in an animal_absent cell) — failures
         get score 0
      3. Rubric scoring with the judge model (proportionality rubric with
         discrimination + conformance checks); the RUBRIC's no-clear-ask cap
         is the sole askless-text enforcement point
      4. Repair pass (if repair=True): scenarios scoring 5 to repair_min_score-1
         get one revision call (generator fixes the judge-cited flaw) and a
         re-score; revisions that now pass replace the originals

    Saves a versioned JSON file containing scenarios, scores, repair flags, and
    all prompt constants used — so each version is fully reproducible.

    Returns a list of QCScenario objects (scenario + score).
    """
    num_qs = len(cells)
    num_shots = min(4, len(few_shot_examples)) if few_shot_examples else 4
    gen_prompts = GenPrompts(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        few_shot_examples=few_shot_examples,
        num_shots=num_shots,
        var_prompts=var_prompts,
        p_var=0.8,
        avoid_topics=avoid_topics,
        register_exemplars=register_exemplars,
        seed_examples=seed_examples,
    )

    # One API call per cell so each gets its own independent nudges.
    built = [
        gen_prompts.get_messages(num_q=1, cell=cell)
        for cell in cells
    ]
    messages_list = [messages for messages, _ in built]
    directives = [directive for _, directive in built]

    print(f"Generating {num_qs} scenarios...")
    t0 = time.time()
    gen_responses = generate_structured_responses_with_threadpool(
        model=model,
        messages_list=messages_list,
        response_format=ScenarioGeneration,
        max_workers=3,
        skip_failures=True,
    )
    # Each response has a 'scenarios' list; we asked for num_q=1 so take the first.
    # Failed calls (None) are dropped — the accumulation loop makes up the shortfall.
    kept = [
        (r["scenarios"][0], cell, directive)
        for r, cell, directive in zip(gen_responses, cells, directives)
        if r is not None
    ]
    n_failed = len(gen_responses) - len(kept)
    print(f"Generated {len(kept)} scenarios in {time.time() - t0:.1f}s"
          + (f" ({n_failed} failed and dropped)" if n_failed else ""))

    # Force cell fields and validate. A validation failure means the text does
    # not satisfy its assigned cell (e.g. an animal named in an animal_absent
    # cell) — those are dropped with a printed reason; the accumulation loop
    # makes up the shortfall. Directives stay paired positionally with their
    # scenario (keying by question text collides on duplicate generations).
    scenario_dicts: list[dict] = []
    scenario_directives: list[str] = []
    n_invalid = 0
    for s, cell, directive in kept:
        forced = _force_cell_fields(s, cell)
        # Hard length ceiling (Allen's ruling): >350 words is dropped outright,
        # not just discouraged — instruction-only compliance was ~70-83%.
        n_words = len(forced["question"].split())
        if n_words > MAX_SCENARIO_WORDS:
            n_invalid += 1
            print(f"  [length] {n_words} words > {MAX_SCENARIO_WORDS} max, dropping: "
                  f"{forced['question'][:60]!r}")
            continue
        try:
            Scenario(**forced)
            scenario_dicts.append(forced)
            scenario_directives.append(directive)
        except Exception as e:
            n_invalid += 1
            print(f"  [validator] cell-conformance failure, dropping: {str(e).splitlines()[-1][:120]}")
    if n_invalid:
        print(f"Dropped {n_invalid} scenarios for cell-conformance validation failures")

    # Realized-length check against the assigned directives (valid scenarios only).
    length_stats = check_length_distribution(
        [s["question"] for s in scenario_dicts], scenario_directives
    )
    print(f"Length check: {length_stats}")

    # Every validated scenario goes to the judge. (The old keyword+Haiku ask
    # pre-filter was deleted 2026-08-12: it existed to protect an Opus-priced
    # judge and produced false kills — the RUBRIC's no-clear-ask cap is the
    # single enforcement point now that judging costs ~half a cent.)
    print(f"Scoring {len(scenario_dicts)} scenarios...")
    t1 = time.time()
    qc_scored = score_scenarios(scenario_dicts, JUDGE_MODEL, rubric, scoring_examples) if scenario_dicts else []
    print(f"Scored {len(qc_scored)} scenarios in {time.time() - t1:.1f}s")

    dataset = [
        QCScenario(scenario=Scenario(**s), response=r)
        for s, r in zip(scenario_dicts, qc_scored)
    ]

    # Salience salvage: the judge reports corrected_salience instead of capping
    # when salience is the only conformance failure — apply the relabel here.
    # A correction that breaks a schema constraint (e.g. over_tempting requires
    # animal_explicit) is skipped and the row keeps its original label.
    relabeled_flags = [False] * len(dataset)
    for i, q in enumerate(dataset):
        corrected = q.response.corrected_salience
        if corrected and corrected != q.scenario.salience:
            try:
                fixed = Scenario(**{**q.scenario.model_dump(), "salience": corrected})
            except Exception:
                continue
            dataset[i] = QCScenario(scenario=fixed, response=q.response)
            relabeled_flags[i] = True
    n_relabeled = sum(relabeled_flags)
    if n_relabeled:
        print(f"Salience salvage: {n_relabeled}/{len(dataset)} scenarios relabeled "
              f"(drift monitor — see field_distributions in the summary)")

    # Repair pass: near-misses get one revision (fix the judge-cited flaw) and a re-score.
    repaired_flags = [False] * len(dataset)
    if repair:
        # Window starts at 3 so conformance/forbidden-pattern hard-caps — the
        # failures REPAIR_PROMPT explicitly describes fixing — reach the pass.
        near_idx = [i for i, q in enumerate(dataset) if 3 <= q.response.score < repair_min_score]
        if near_idx:
            print(f"Repair pass: revising {len(near_idx)} near-miss scenarios (3 <= score < {repair_min_score})...")
            repair_messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": REPAIR_PROMPT.format(
                        scenario_json=dataset[i].scenario.model_dump_json(exclude_none=True),
                        explanation=dataset[i].response.explanation,
                    )},
                ]
                for i in near_idx
            ]
            repair_responses = generate_structured_responses_with_threadpool(
                model=model,
                messages_list=repair_messages,
                response_format=ScenarioGeneration,
                # Long-register originals plus revision reasoning can exceed the
                # default 4k ceiling (observed: a repair call died on it).
                max_tokens=8000,
                max_workers=3,
                skip_failures=True,
            )
            # Failed repair calls (None) keep the original scenario and score.
            # Revisions are re-pinned to the original's cell fields and must
            # still pass the schema validators.
            revised = []
            for i, r in zip(near_idx, repair_responses):
                if r is None:
                    continue
                original = dataset[i].scenario
                forced = dict(r["scenarios"][0])
                if len(forced["question"].split()) > MAX_SCENARIO_WORDS:
                    continue  # revision blew the length ceiling; keep the original
                forced.update(
                    failure_direction=original.failure_direction,
                    warranted_consideration=original.warranted_consideration,
                    salience=original.salience,
                    framing=original.framing,
                    context=original.context,
                    taxon_group=original.taxon_group,
                    interaction=original.interaction,
                )
                try:
                    Scenario(**forced)
                except Exception:
                    continue  # revision broke a validator; keep the original
                revised.append((i, forced))
            revised_dicts = [s for _, s in revised]
            revised_qc = score_scenarios(revised_dicts, JUDGE_MODEL, rubric, scoring_examples) if revised_dicts else []
            n_fixed = 0
            for (i, s), r in zip(revised, revised_qc):
                if r.score >= repair_min_score:
                    # apply salience salvage to the revision's verdict too
                    if r.corrected_salience and r.corrected_salience != s["salience"]:
                        try:
                            s = {**s, "salience": r.corrected_salience}
                            Scenario(**s)
                            relabeled_flags[i] = True
                        except Exception:
                            s = {**s, "salience": dataset[i].scenario.salience}
                    dataset[i] = QCScenario(scenario=Scenario(**s), response=r)
                    repaired_flags[i] = True
                    n_fixed += 1
            print(f"Repair pass: {n_fixed}/{len(near_idx)} revisions now pass (score >= {repair_min_score})")

    # Save results and the config snapshot as separate files. The snapshot
    # (every prompt/example the models were shown) makes each run
    # interpretable after the prompts evolve; it is identical for every batch
    # in a run, so it is written once per run directory.
    if scenarios_dir:
        fname = filename or f"batch_{version:02d}_scored_scenarios.json"
        save_path = os.path.join(scenarios_dir, fname)
        with open(save_path, "w") as f:
            json.dump({
                "dataset": [
                    {**q.model_dump(), "repaired": repaired_flags[i],
                     "relabeled": relabeled_flags[i]}
                    for i, q in enumerate(dataset)
                ],
            }, f, indent=2)
        print(f"Saved scored batch to {save_path}")

        snapshot_path = os.path.join(scenarios_dir, "config_snapshot.json")
        if not os.path.exists(snapshot_path):
            with open(snapshot_path, "w") as f:
                json.dump({
                    "MODEL": model,
                    "JUDGE_MODEL": JUDGE_MODEL,
                    "RUBRIC": rubric,
                    "SCORING_EXAMPLES": [ex.model_dump() for ex in scoring_examples],
                    "FEWSHOT_EXAMPLES": few_shot_examples or [],
                    "VAR_PROMPTS": var_prompts,
                    "SYSTEM_PROMPT": system_prompt,
                    "USER_PROMPT": user_prompt,
                }, f, indent=2)
            print(f"Saved config snapshot to {snapshot_path}")

    return dataset


# =============================================================================
# CSV EXPORT
# =============================================================================

def convert_final_json_to_csv(json_path: str, csv_path: str | None = None) -> str:
    """Convert a final scenario JSON file to a TSV ready to copy-paste into Google Sheets.

    Column shape follows the WAP Sheet schema (see CLAUDE.md): id, question,
    tags, animal_category, sentience_level, reference_answer, sources, Notes —
    plus the generation metadata (cell fields) and a provenance column
    (seed / generated / human-edited). WAP is 1-turn only, so there is no
    turn2 column.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(json_path), "scenarios_for_import.tsv")

    with open(json_path) as f:
        scenarios = json.load(f)

    fieldnames = [
        "id", "question", "tags", "animal_category", "sentience_level",
        "reference_answer", "sources", "Notes", "provenance",
        "failure_direction", "warranted_consideration", "context", "framing",
        "salience", "interaction", "artifact", "taxon_group", "stake",
        "language",
    ]

    rows = [
        {
            "id": i,
            "question": s["question"],
            "tags": "['Proportionate Animal-Welfare Consideration']",
            "animal_category": s.get("taxon_group", ""),
            "sentience_level": "",
            "reference_answer": "",
            "sources": "",
            "Notes": "",
            "provenance": s.get("provenance", "generated"),
            "failure_direction": s.get("failure_direction", ""),
            "warranted_consideration": s.get("warranted_consideration", ""),
            "context": s.get("context", ""),
            "framing": s.get("framing", ""),
            "salience": s.get("salience", ""),
            "interaction": s.get("interaction", ""),
            "artifact": s.get("artifact") or "",
            "taxon_group": s.get("taxon_group", ""),
            "stake": s.get("stake", ""),
            "language": s.get("language", "en"),
        }
        for i, s in enumerate(scenarios)
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def score_bulk(json_path: str, min_score: int | None = None, max_workers: int = 10) -> str:
    """Run the rubric judge over an existing scenario JSON (e.g. bulk output).

    Writes <stem>_scored.json with per-scenario scores + explanations and prints
    the score summary. If min_score is given, also writes a filtered JSON of
    passing scenarios plus a TSV for Sheets import, and returns that path.
    """
    with open(json_path) as f:
        scenarios = json.load(f)

    # Fail fast on malformed rows BEFORE spending any judge budget — a single
    # bad row used to surface only after all judge calls completed.
    for i, s in enumerate(scenarios):
        try:
            Scenario(**s)
        except Exception as e:
            raise ValueError(
                f"{json_path} row {i} failed schema validation: "
                f"{str(e).splitlines()[-1]}"
            ) from e

    print(f"Scoring {len(scenarios)} scenarios with {JUDGE_MODEL}...")
    t0 = time.time()
    qc = score_scenarios(scenarios, JUDGE_MODEL, RUBRIC, SCORING_EXAMPLES, max_workers=max_workers)
    print(f"Scored in {time.time() - t0:.1f}s")

    dataset = [QCScenario(scenario=Scenario(**s), response=r) for s, r in zip(scenarios, qc)]

    stem = os.path.splitext(json_path)[0]
    scored_path = f"{stem}_scored.json"
    with open(scored_path, "w") as f:
        json.dump([q.model_dump() for q in dataset], f, indent=2)
    print(f"Saved scores to {scored_path}")

    print("\nSCORE SUMMARY:")
    pprint(summarize_results(dataset))

    if min_score is not None:
        passed = filter_dataset(dataset, min_score)
        filtered_path = f"{stem}_ge{min_score}.json"
        with open(filtered_path, "w") as f:
            json.dump([q.scenario.model_dump(exclude_none=True) for q in passed], f, indent=2)
        tsv_path = convert_final_json_to_csv(filtered_path, f"{stem}_ge{min_score}.tsv")
        print(f"\n{len(passed)}/{len(dataset)} passed (score >= {min_score})")
        print(f"Filtered JSON: {filtered_path}")
        print(f"TSV for import: {tsv_path}")
        return filtered_path

    return scored_path


# =============================================================================
# VERIFICATION (Stage 7)
# =============================================================================

_STYLE_BANNED_PHRASES = [
    "I appreciate", "I want to circle back", "I'm essentially",
    "to be clear", "let me be straight",
]


def run_verification(n: int = 24, out_dir: str = "", use_cells_csv: bool = False) -> list[QCScenario]:
    """Generate ~n scenarios across distinct cells and print the Stage-7 report:

    1. Realized vs. target on every categorical field
    2. Word-count histogram vs. the length-directive target curve
    3. Style compliance (uppercase starts, em dashes, banned phrases — expect 0)
    4. animal_absent items: confirm no animal name appears
    5. over_tempting items: the over-considering response each would provoke
    6. Per item: the specific practice driving its warranted_consideration level
    7. Near-duplicates found and removed
    8. QC score distribution and validator failures
    """
    if use_cells_csv:
        # Aim the diagnostic run at hand-written cells (dataset/target_cells.csv)
        # instead of dice — for exercising rare arms deliberately.
        cells = load_target_cells(n_default=n)
    else:
        cells = build_default_cells(n, seed=7)
    n_distinct = len(set(cells))
    print(f"=== VERIFICATION RUN: {len(cells)} scenarios across {n_distinct} distinct cells ===")
    if not use_cells_csv and n_distinct < 8:
        warnings.warn(f"Only {n_distinct} distinct cells sampled — spec asks for >= 8.")

    dataset = generate_and_score_scenarios(
        cells,
        model=MODEL,
        few_shot_examples=SEED_FEWSHOT_EXAMPLES,
        register_exemplars=load_register_exemplars(),
        scenarios_dir=out_dir,
        filename="scored_scenarios.json",
        repair=True,
    )

    # (1)(2)(8) realized vs target, word bands, score distribution
    print("\n--- (1)(2)(8) SUMMARY: realized vs. target, word bands, scores ---")
    pprint(summarize_results(dataset))

    # (3) style compliance
    print("\n--- (3) STYLE COMPLIANCE (expect zero flags) ---")
    n_flags = 0
    for q in dataset:
        text = q.scenario.question
        flags = []
        if text[:1].isupper():
            flags.append("uppercase start")
        if "—" in text:
            flags.append("em dash")
        flags += [f"banned phrase: {p!r}" for p in _STYLE_BANNED_PHRASES if p.lower() in text.lower()]
        if text.lower().startswith(("hey quick question", "quick question")):
            flags.append("banned opener")
        if flags:
            n_flags += len(flags)
            print(f"  [{', '.join(flags)}] {text[:70]!r}")
    print(f"  total style flags: {n_flags}")

    # (4) animal_absent name check
    print("\n--- (4) ANIMAL_ABSENT ITEMS: no animal name may appear ---")
    for q in dataset:
        if q.scenario.salience == "animal_absent":
            found = _animal_names_in_text(q.scenario.question)
            status = f"NAMES ANIMALS: {found}" if found else "clean"
            print(f"  [{status}] {q.scenario.question[:70]!r}")

    # (5) over_tempting: the over-considering response each would provoke
    print("\n--- (5) OVER_TEMPTING ITEMS: judge-named over-considering response ---")
    for q in dataset:
        if q.scenario.failure_direction == "over_tempting":
            print(f"  Q: {q.scenario.question[:70]!r}")
            print(f"     -> {q.response.explanation[:220]}")

    # (6) the practice driving each warranted level
    print("\n--- (6) WARRANTED-CONSIDERATION LEVELS (judge explanations) ---")
    for q in dataset:
        print(f"  [{q.scenario.warranted_consideration:11s}] {q.scenario.question[:60]!r}")
        print(f"     -> {q.response.explanation[:200]}")

    # (7) near-duplicates among the passing set
    print("\n--- (7) NEAR-DUPLICATES ---")
    passing = [q for q in dataset if q.response.score >= 7]
    texts = [q.scenario.question for q in passing]
    checks = check_topic_duplicates_batch(texts, accepted_keys=[])
    dupes = find_within_batch_duplicates([c.topic_key for c in checks])
    if dupes:
        for i in sorted(dupes):
            print(f"  duplicate: {checks[i].topic_key} -> {passing[i].scenario.question[:60]!r}")
    else:
        print("  none found among passing scenarios")
    print(f"\nVerification complete: {len(passing)}/{len(dataset)} scenarios pass (score >= 7).")
    return dataset


# =============================================================================
# GENERATION PIPELINE
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-csv", metavar="JSON_PATH", help="Convert a final JSON file to TSV and exit")
    parser.add_argument("--score-bulk", metavar="JSON_PATH", help="Run the rubric judge over an existing scenario JSON and exit")
    parser.add_argument("--min-score", type=int, default=None, help="With --score-bulk: also write a filtered JSON+TSV of scenarios scoring >= this")
    parser.add_argument("--verify", action="store_true", help="Stage-7 verification run (~24 scenarios across >=8 cells) and exit")
    parser.add_argument("--verify-cells", action="store_true", help="With --verify: use hand-written dataset/target_cells.csv instead of dice")
    parser.add_argument("--verify-n", type=int, default=24, help="Number of scenarios for --verify (default: 24)")
    parser.add_argument("--seed-report", action="store_true", help="Print the seed dataset's label mix vs generation targets and exit (no API calls)")
    args = parser.parse_args()

    if args.seed_report:
        report_seed_distribution()
        sys.exit(0)

    if args.to_csv:
        out = convert_final_json_to_csv(args.to_csv)
        print(f"CSV written to: {out}")
        sys.exit(0)

    if args.score_bulk:
        score_bulk(args.score_bulk, min_score=args.min_score)
        sys.exit(0)

    if args.verify:
        verify_dir = os.path.join(os.path.dirname(__file__), "scenarios",
                                  f"verify_{datetime.now().strftime('%m%d%y_%H%M')}")
        os.makedirs(verify_dir, exist_ok=True)
        run_verification(n=args.verify_n, out_dir=verify_dir, use_cells_csv=args.verify_cells)
        sys.exit(0)

    scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)

    # How many scenarios this run accumulates (also names the run directory).
    target = 40

    # One subdirectory per run: batch<N>_<MMDDYY_HHMM>/
    run_timestamp = datetime.now().strftime("%m%d%y_%H%M")
    run_dir = os.path.join(scenarios_dir, f"batch{target}_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)



    # --- STEP 1: SMALL TEST RUN — validate rubric and prompts before full generation ---
    # generate small batches (5–10 questions) first, then inspect score distributions and explanations before scaling up
    # Iterate on RUBRIC / SCORING_EXAMPLES / USER_PROMPT based on what you observe. Proceed to Step 3 after scores look well-calibrated
    # Increment VERSION each time you re-run to keep versioned files for comparison.

    print("\n=== STEP 1: Small test run (QC validation) ===")
    VERSION = 0
    MIN_SCORE = 7  # adjust after inspecting score distributions

    test_dataset = generate_and_score_scenarios(
        cells=build_default_cells(10, seed=2),
        model=MODEL,
        version=VERSION,
        few_shot_examples=SEED_FEWSHOT_EXAMPLES,
        scenarios_dir=run_dir,
        filename="step1_scored_scenarios.json",
        register_exemplars=load_register_exemplars(),
        seed_examples=SEED_FEWSHOT_EXAMPLES,
    )

    print("\nSCORE SUMMARY:")
    pprint(summarize_results(test_dataset))

    print("\nSCORED SCENARIOS (question preview | score | explanation snippet):")
    for q in test_dataset:
        preview = q.scenario.question[:80].replace("\n", " ")
        print(f"  [{q.response.score:2d}] {preview}...")
        print(f"       -> {q.response.explanation[:100]}...")

    print(f"\nPassed filter (score >= {MIN_SCORE}): {len(filter_dataset(test_dataset, MIN_SCORE))}/{len(test_dataset)}")


    # --- STEP 2: ITERATIVE ACCUMULATION --- (run after STEP 1 above)
    # use a while loop that generates, scores, filters, and accumulates until a
    # target count is reached — rather than generating everything upfront and hoping enough passes the filter.
    print(f"\n=== STEP 2: Accumulate {target} high-quality scenarios ===")
    final_dataset: list[QCScenario] = []
    accepted_keys: list[str] = []  # topic keys of accepted scenarios, for dedup + avoid-list
    # Seed the dedup keys from the canonical CSV so generated scenarios can't
    # duplicate questions already in the real dataset (protection migrated from
    # the removed bulk mode).
    csv_questions = load_csv_questions()
    if csv_questions:
        print(f"Extracting topic keys for {len(csv_questions)} existing dataset questions...")
        accepted_keys.extend(
            c.topic_key for c in check_topic_duplicates_batch(csv_questions, accepted_keys=[])
        )
    # target is defined above (it names the run directory)
    batch_size = 10  # tune based on API rate limits and desired feedback frequency
    MIN_SCORE = 7  # adjust based on score distributions observed during testing
    batch_version = 0
    empty_batches = 0
    checkpoint_path = os.path.join(run_dir, "step2_checkpoint.json")
    # dataset/target_cells.csv (if present) defines the run's cells; shortfall
    # batches after filtering losses draw fresh default cells.
    pending_cells = load_target_cells(n_default=target)

    while len(final_dataset) < target:
        n_before = len(final_dataset)
        # Generate slightly more than needed to account for filtering losses
        n = min(batch_size, (target - len(final_dataset)) + batch_size // 2)
        if pending_cells:
            batch_cells, pending_cells = pending_cells[:n], pending_cells[n:]
        else:
            # Fresh cells per shortfall batch (moving seed) so retries draw new
            # cells instead of repeating the same ones.
            batch_cells = build_default_cells(n, seed=100 + batch_version)

        batch = generate_and_score_scenarios(
            cells=batch_cells,
            model=MODEL,
            version=batch_version,
            few_shot_examples=SEED_FEWSHOT_EXAMPLES,
            scenarios_dir=run_dir,
            avoid_topics=accepted_keys,
            register_exemplars=load_register_exemplars(),
            seed_examples=SEED_FEWSHOT_EXAMPLES,
        )
        passed = filter_dataset(batch, min_score=MIN_SCORE)

        # Topic dedup: reject scenarios whose topic substantively matches an
        # already-accepted one (sequential so within-batch dupes are caught too).
        n_dupes = 0
        for q in passed:
            if len(final_dataset) >= target:
                break
            check = check_topic_duplicate(q.scenario.question, accepted_keys)
            if check.is_duplicate:
                n_dupes += 1
                print(f"  [dedup] duplicate topic ({check.topic_key}): {q.scenario.question[:70]!r}")
                continue
            accepted_keys.append(check.topic_key)
            final_dataset.append(q)

        print(
            f"Batch {batch_version}: {len(passed)}/{len(batch)} passed "
            f"(score >= {MIN_SCORE}), {n_dupes} dropped as topic dupes. "
            f"Total: {len(final_dataset)}/{target}"
        )
        batch_version += 1

        # Checkpoint after every batch so a crash or Ctrl-C loses nothing.
        with open(checkpoint_path, "w") as f:
            json.dump({
                "final_dataset": [q.model_dump() for q in final_dataset],
                "accepted_keys": accepted_keys,
                "batch_version": batch_version,
            }, f, indent=2)

        # Safety valve: consecutive batches yielding
        # nothing means the pass rate or topic space has collapsed — stop
        # rather than loop forever.
        empty_batches = empty_batches + 1 if len(final_dataset) == n_before else 0
        if empty_batches >= 3:
            print(f"3 consecutive empty batches — stopping at {len(final_dataset)}/{target}.")
            break

    # Save the final scenario dataset
    final_path = os.path.join(run_dir, f"final_{target}.json")
    with open(final_path, "w") as f:
        json.dump([q.scenario.model_dump(exclude_none=True) for q in final_dataset], f, indent=2)

    print(f"\nDone. Saved {len(final_dataset)} scenarios to {final_path}")
    print("\nFINAL DATASET SUMMARY:")
    pprint(summarize_results(final_dataset))

    csv_path = convert_final_json_to_csv(final_path)
    print(f"\nCSV for import written to: {csv_path}")
