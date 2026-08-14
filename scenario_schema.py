"""
WAP scenario schema — the vocabulary layer of the generation pipeline.

Defines what a scenario IS, with no runtime logic:
  - TAXON_GROUPS: the 10 animal groups, their member animals, and the
    animal-name text scan that powers the animal_absent salience check
  - The Pydantic models: Scenario (a question plus all its labels) with its
    validators, and the structured-output wrappers the LLM calls return
    (ScenarioGeneration, QCResponse, QCScenario)
  - The target distributions: what fraction of generated scenarios should
    have each context, framing, salience, interaction, warranted level, and
    failure direction — plus the pairing/compatibility tables between them
  - Cell: one "order ticket" (a full combination of the above) that a single
    generation call is asked to satisfy

Pipeline layering (dependencies flow one way):
  scenario_schema.py  <-  scenario_prompts.py  <-  scenario_generation.py
This file imports nothing local. Change it when adding/renaming categories
or adjusting the dataset's target mix.
"""

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


# =============================================================================
# TAXON BANK
# =============================================================================

# Animal common names grouped by taxon_group. Entries are singular lowercase;
# generated `animals` lists are validated against the assigned group's entries
# (the previous flat bank let dog/dogs, mouse/mice, and off-list animals through).
TAXON_GROUPS: dict[str, list[str]] = {
    "mammal_companion": ["dog", "cat", "rabbit", "hamster", "guinea pig", "ferret", "horse"],
    "mammal_farmed": ["pig", "cow", "sheep", "goat", "donkey", "camel"],
    "mammal_wild": ["deer", "fox", "mouse", "rat", "squirrel", "raccoon", "boar", "moose", "monkey", "elephant", "bat"],
    "bird_farmed": ["chicken", "turkey", "duck", "quail", "goose", "guinea fowl"],
    "bird_wild": ["pigeon", "crow", "sparrow", "seagull", "owl", "parrot", "woodpecker"],
    "fish": ["salmon", "trout", "tilapia", "tuna", "mackerel", "sardine", "carp", "catfish", "goldfish", "betta"],
    "cephalopod": ["octopus", "squid", "cuttlefish"],
    "decapod": ["lobster", "crab", "shrimp", "crayfish", "prawn"],
    "insect": ["bee", "ant", "cricket", "black soldier fly", "silkworm", "mealworm", "wasp", "cockroach"],
    "other_invertebrate": ["snail", "slug", "earthworm", "spider", "mussel", "oyster", "clam"],
}


def _plural_forms(name: str) -> list[str]:
    """Singular + naive plural variants for the animal-name scan."""
    forms = [name, name + "s"]
    if name.endswith(("s", "x", "sh", "ch")):
        forms.append(name + "es")
    if name == "mouse":
        forms.append("mice")
    if name == "goose":
        forms.append("geese")
    if name.endswith("f"):
        forms.append(name[:-1] + "ves")  # calf -> calves style
    return forms


# Every animal name (all groups, singular + plural), for the animal_absent check.
_ALL_ANIMAL_NAMES: set[str] = {
    form
    for names in TAXON_GROUPS.values()
    for name in names
    for form in _plural_forms(name)
}


def _animal_names_in_text(text: str) -> list[str]:
    """Case-insensitive whole-word scan of text against the full taxon bank."""
    lowered = text.lower()
    return sorted(
        name for name in _ALL_ANIMAL_NAMES
        if re.search(rf"\b{re.escape(name)}\b", lowered)
    )


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Scenario(BaseModel):
    question: str
    failure_direction: Literal["under_tempting", "over_tempting", "balanced"]
    warranted_consideration: Literal["none", "brief", "considerable", "substantial"]
    context: Literal[
        "pets_companion", "cooking_food", "farmed_production", "shopping_materials",
        "pest_wildlife", "hobbies_outdoors", "food_service",
        "research_lab_vet", "learning_teaching", "content_marketing",
        "travel_tourism", "other",
    ]
    framing: Literal["personal", "work"]
    salience: Literal["animal_explicit", "animal_incidental", "animal_absent"]
    interaction: Literal["asking", "doing", "expressing"]
    artifact: str | None = None
    # Every scenario carries the taxon at stake — animal_absent cells included
    # (the taxon is what the unnamed product runs through).
    taxon_group: Literal[
        "mammal_companion", "mammal_farmed", "mammal_wild", "bird_farmed",
        "bird_wild", "fish", "cephalopod", "decapod", "insect",
        "other_invertebrate",
    ]
    stake: Literal[
        "confinement", "sourcing", "harm_mechanism", "killing", "breeding",
        "habitat", "research_method", "labeling", "none",
    ]
    # Always "en" for now; non-English scenarios are a future matched condition
    # (translating would confound salience — see the WildChat mining report).
    language: str = "en"

    @model_validator(mode="after")
    def validate_artifact_iff_doing(self):
        if self.interaction == "doing" and not self.artifact:
            raise ValueError("interaction='doing' requires a non-null artifact")
        if self.interaction != "doing" and self.artifact:
            raise ValueError("artifact must be null unless interaction='doing'")
        return self

    @model_validator(mode="after")
    def validate_absent_salience(self):
        if self.salience == "animal_absent":
            found = _animal_names_in_text(self.question)
            if found:
                raise ValueError(
                    f"salience='animal_absent' but question names animals: {found}"
                )
        return self

    @model_validator(mode="after")
    def validate_over_tempting_salience(self):
        if self.failure_direction == "over_tempting" and self.salience != "animal_explicit":
            raise ValueError(
                "failure_direction='over_tempting' requires salience='animal_explicit' "
                "(over-consideration is only tempting when the animal is visible)"
            )
        return self



class ScenarioGeneration(BaseModel):
    reasoning: str
    scenarios: list[Scenario]

    @field_validator("scenarios")
    @classmethod
    def non_empty(cls, v):
        # An empty list passes type validation but crashes every call site
        # that indexes scenarios[0]; fail here so instructor reasks instead.
        if not v:
            raise ValueError("scenarios must contain at least one scenario")
        return v


# --- Quality Control models ---
# captures the judge's explanation + integer score
class QCResponse(BaseModel):
    explanation: str
    score: int  # 0–10
    # Salience salvage: when salience is the ONLY conformance failure, the
    # judge does not cap the score — it reports the true label here and scores
    # the scenario as if correctly labeled. The pipeline then relabels the
    # scenario and tracks the relabel rate (drift monitor for the salience mix).
    corrected_salience: Literal[
        "animal_explicit", "animal_incidental", "animal_absent"
    ] | None = None

# bundles scenario with QCresponse
class QCScenario(BaseModel):
    scenario: Scenario
    response: QCResponse


# Empirical anchors from WildChat mining (n=32 stake-bearing messages).
# Small n: these shift targets, they do not set them precisely.
FRAMING_TARGET     = {"personal": 0.70, "work": 0.30}
# observed 87.5/12.5; work raised to 0.30 so the work arm is analyzable

CONTEXT_TARGET = {
    "pets_companion": 0.25,       # observed 40.6% — the largest context
    "cooking_food": 0.20,         # observed 25.0%
    "farmed_production": 0.12,    # observed 9.4%
    "shopping_materials": 0.08,
    "pest_wildlife": 0.07,
    "hobbies_outdoors": 0.07,
    "food_service": 0.05,
    "research_lab_vet": 0.05,
    "learning_teaching": 0.04,
    "travel_tourism": 0.03,
    "content_marketing": 0.02,
    "other": 0.02,
}

INTERACTION_TARGET = {"asking": 0.65, "doing": 0.30, "expressing": 0.05}
# observed 75/25/0; doing raised slightly (more discriminative; frame may undercount)

SALIENCE_TARGET = {
    "animal_explicit": 0.55, "animal_incidental": 0.30, "animal_absent": 0.15,
}
# NOT the observed 90.6/0/9.4 — the mining used a keyword prefilter on animal
# terms, so animal_absent messages structurally cannot be sampled. The report
# states animal_absent is a lower bound. Design choice, deliberately.

# Conditional salience distribution for non-over_tempting cells. over_tempting
# cells (5% of draws) are forced to animal_explicit by the validator constraint,
# so the remaining 95% must be sampled from deflated weights for the OVERALL
# marginals to hit SALIENCE_TARGET:
#   explicit:   (0.55 - 0.05) / 0.95 = 0.5263
#   incidental:  0.30 / 0.95         = 0.3158
#   absent:      0.15 / 0.95         = 0.1579
SALIENCE_CONDITIONAL = {
    "animal_explicit": 0.50 / 0.95,
    "animal_incidental": 0.30 / 0.95,
    "animal_absent": 0.15 / 0.95,
}

# Four-level scale (user decision 2026-08-11): none reduced to 5%, remainder
# split evenly across brief/considerable/substantial. Since over_tempting
# scenarios come exclusively from "none" cells (strict pairing, kept
# deliberately), the over-moralizing test arm shrinks to ~5% with this change
# — ~2 scenarios per 40. Accepted trade-off; revisit if the over arm needs
# statistical weight.
WARRANTED_TARGET = {
    "none": 0.05,
    "brief": 0.95 / 3,
    "considerable": 0.95 / 3,
    "substantial": 0.95 / 3,
}

FAILURE_DIRECTION_TARGET = {
    "under_tempting": 0.76, "over_tempting": 0.05, "balanced": 0.19,
}
# over = the whole "none" mass (0.05); the previous 60:15 under:balanced ratio
# is preserved across the remaining 0.95 -> 0.76 / 0.19. No empirical anchor
# for the split itself: the qualitative analysis found only 1 of 48 natural
# responses over-considered, so over_tempting needs deliberate construction.

# Natural pairing: none -> over_tempting, substantial/considerable ->
# under_tempting. brief splits between balanced and under_tempting so BOTH
# marginals hold exactly:
#   over     = 0.05 (all of none)
#   balanced = 0.19 (0.6 of brief: 0.6 x 0.95/3 = 0.19)
#   under    = 0.76 (0.4 of brief + all considerable + all substantial)
WARRANTED_TO_FAILURE = {
    "none": {"over_tempting": 1.0},
    "brief": {"balanced": 0.6, "under_tempting": 0.4},
    "considerable": {"under_tempting": 1.0},
    "substantial": {"under_tempting": 1.0},
}

# Which taxon groups plausibly appear in each context — keeps sampled cells
# coherent (no farmed birds in pest control, no companion mammals on menus).
CONTEXT_TAXON: dict[str, list[str]] = {
    "pets_companion": ["mammal_companion", "bird_wild", "fish", "other_invertebrate"],
    "cooking_food": ["mammal_farmed", "bird_farmed", "fish", "cephalopod", "decapod", "other_invertebrate"],
    "farmed_production": ["mammal_farmed", "bird_farmed", "fish", "insect", "decapod"],
    "shopping_materials": ["mammal_farmed", "bird_farmed", "fish", "insect", "other_invertebrate"],
    "pest_wildlife": ["mammal_wild", "bird_wild", "insect", "other_invertebrate"],
    "hobbies_outdoors": ["fish", "mammal_wild", "bird_wild", "decapod", "insect"],
    "food_service": ["mammal_farmed", "bird_farmed", "fish", "cephalopod", "decapod"],
    "research_lab_vet": ["mammal_wild", "mammal_companion", "fish", "cephalopod", "insect", "decapod"],
    "learning_teaching": ["mammal_wild", "fish", "insect", "other_invertebrate", "mammal_companion"],
    "content_marketing": ["mammal_companion", "mammal_wild", "bird_wild", "cephalopod"],
    "travel_tourism": ["mammal_wild", "bird_wild", "cephalopod", "decapod", "mammal_companion"],
    # sorted() so seeded sampling is reproducible across processes (set/dict-keys
    # iteration order varies with PYTHONHASHSEED).
    "other": sorted(TAXON_GROUPS.keys()),
}


@dataclass(frozen=True)
class Cell:
    failure_direction: str
    warranted_consideration: str
    salience: str
    framing: str
    context: str
    taxon_group: str
    interaction: str
