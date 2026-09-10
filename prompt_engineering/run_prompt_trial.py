#!/usr/bin/env python3
"""Run prompt-engineering trials on the high-agreement RNL subset.

The script intentionally keeps production prompts untouched. It creates the
agreed subset from the updated agreement CSV, samples a balanced 30-row dev set,
runs candidate prompt templates through the existing LLM client/parser/filter,
and scores generated KB columns with the existing multi-reference F1 logic.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calculate_multi_reference_f1 import (
    EvaluationResult,
    evaluate_prediction_column,
    format_score,
    parse_kb_cell,
    write_summary,
)
from kbprojection.filtering import pipeline_filter_kb_injections
from kbprojection.llm import AsyncGenericAIClient, _extract_validated_kb_from_output
from kbprojection.models import NLILabel, NLIProblem
from kbprojection.prompts import ETTORE_BASE_PROMPT, LASHA_BASE_PROMPT


DEFAULT_REFERENCE_COLUMNS = [
    "Alternative_KB",
    "Ettore_KB",
    "Jorryt_KB",
    "Lasha_KB",
    "Stefan_KB",
]
DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-oss-20b",
]


@dataclass(frozen=True)
class CandidatePrompt:
    name: str
    description: str
    template: str


@dataclass(frozen=True)
class GeneratedColumns:
    raw_response: str
    kb: str
    error: str


def insert_before(template: str, marker: str, addition: str) -> str:
    if marker not in template:
        raise ValueError(f"Marker not found in template: {marker}")
    return template.replace(marker, f"{addition.rstrip()}\n\n{marker}", 1)


ETTORE_FINAL_QUERY_MARKER = "Premise: ${premise}"
LASHA_FINAL_QUERY_MARKER = (
    "Now process the following input while strictly following the above instructions and formatting."
)

ETTORE_PLUS_PRECISION = insert_before(
    ETTORE_BASE_PROMPT,
    ETTORE_FINAL_QUERY_MARKER,
    """## CALIBRATION UPDATE: precision without changing the task

Keep all rules above. Apply these extra checks only to resolve ambiguity:
- Prefer no relation over a broad event or social inference.
- Do not infer activities from objects or situations, for example carrying a shopping bag is not enough for shop.
- Do not infer event paraphrases unless the two verbs are direct lexical paraphrases.
- Output at most one form of the same bridge; never output both an inflected phrase and its lemma.
- If a candidate relation only proves a sentence that is already syntactically entailed, omit it.
""",
)

ETTORE_PLUS_PRECISION_SOFT = insert_before(
    ETTORE_BASE_PROMPT,
    ETTORE_FINAL_QUERY_MARKER,
    """## CALIBRATION UPDATE: precise and sufficient KB relations

Keep all rules, examples, predicates, and formatting requirements above.
Use the following checks only to resolve ambiguous cases:

1. Output a relation only when it bridges a real lexical mismatch that is needed by the symbolic proof.
2. Check the direction carefully: the premise argument must entail, be a type of, or be a direct paraphrase of the hypothesis argument.
3. Retain direct synonyms, hypernyms, category relations, and valid event entailments. For example, sprint -> run may be valid because sprinting is a type of running.
4. Reject relations based on likely intentions, associated objects, broad situations, social expectations, or possible consequences. For example, carrying a shopping bag does not by itself entail shopping.
5. Reject modifier deletion, morphology-only differences, and relations that merely restate an entailment already established by syntax.
6. Use the shortest correct argument pair. Do not output whole clauses when a word or compact phrase is sufficient.
7. Output only one canonical form of each bridge. Never output both an inflected form and its lemma.
8. After selecting the relations, remove every relation that is not needed for the proof.
9. If no relation passes these checks, output an empty KB.

Perform these checks internally. Output only the final KB in the original required format.
""",
)

ETTORE_PLUS_LEMMA_CANONICAL = insert_before(
    ETTORE_BASE_PROMPT,
    ETTORE_FINAL_QUERY_MARKER,
    """## CALIBRATION UPDATE: canonical relation form

Keep all rules above. When a relation is needed:
- Choose the shortest correct argument pair.
- For ordinary verbs, output the lemma form only, not the inflected surface form.
- For adjectives and nouns, keep the surface word if it is already the natural lexical item.
- Do not output duplicate variants of the same bridge.
- Do not output whole clauses as arguments.
- If the only possible relation would be modifier deletion or morphology-only, output an empty KB.
""",
)

ETTORE_PLUS_HUMAN_ANNOTATION = insert_before(
    ETTORE_BASE_PROMPT,
    ETTORE_FINAL_QUERY_MARKER,
    """## CALIBRATION UPDATE: match the human KB annotation style

Keep all rules above. The desired output should look like a compact human KB annotation:
- Usually zero, one, or two relations.
- Each argument should be the minimal differing lexical item or compact noun phrase.
- Good relation types include synonym, paraphrase, person/category, object/category, adjective/state, location paraphrase, and clear opposition.
- Bad relations include loose activity implications, modifier dropping, syntactic restatements, and duplicate lemma/surface variants.
- If reasonable human annotators would likely leave the KB empty, output an empty KB.
""",
)

LASHA_PLUS_PRECISION = insert_before(
    LASHA_BASE_PROMPT,
    LASHA_FINAL_QUERY_MARKER,
    """Additional calibration while preserving all rules above:
- Only output lexical entailment relations that are both factually acceptable and needed to explain the entailment.
- Do not output relations for entailments that follow without a non-trivial lexical bridge.
- Do not use event or social implications as lexical entailment unless the phrase relation is a direct paraphrase.
- Do not output modifier-dropping relations such as old woman -> woman or military men -> men.
- Do not output both an inflected form and a lemmatized form for the same relation.
- If the best relation would violate any existing formatting rule, omit it instead of approximating it.
""",
)

LASHA_PLUS_PRECISION_RECOVERY = insert_before(
    LASHA_BASE_PROMPT,
    LASHA_FINAL_QUERY_MARKER,
    """Additional calibration while preserving all rules above:

Before producing the final answer, evaluate every candidate relation using the following checks:

1. The relation must bridge a real lexical difference between the premise and the hypothesis.
2. The relation must be necessary to establish the entailment, rather than merely describing something generally true.
3. The direction must be correct: the premise expression must entail or be a direct paraphrase of the hypothesis expression.
4. Use the shortest lexical items or phrases that preserve the intended meaning.
5. Do not output relations based only on modifier deletion, grammatical variation, tense, number, auxiliaries, or word order.
6. Do not infer an activity, intention, consequence, or social situation from an associated object or context.
7. A verb or event relation is allowed when the expressions are direct lexical paraphrases or when the premise verb clearly entails the hypothesis verb, such as sprint -> run. Indirect implications such as carrying a shopping bag -> shopping are not allowed.
8. Do not output both an inflected form and a lemmatized form of the same relation.
9. Output the smallest sufficient set of relations. If no candidate passes every check, output an empty relation set.

Do not include these checks or any explanation in the final answer. Keep the original answer and relations format exactly.
""",
)

LASHA_PLUS_MINIMAL_SCHEMA = insert_before(
    LASHA_BASE_PROMPT,
    LASHA_FINAL_QUERY_MARKER,
    """Additional calibration while preserving all rules above:
- Keep the original answer/relations format exactly.
- Make the relations set minimal: every relation must correspond to one necessary lexical mismatch.
- Prefer the shortest acceptable lemmatized phrase from the premise and the shortest acceptable lemmatized phrase from the hypothesis.
- Avoid whole-clause or long event descriptions.
- Use an empty relation set when the premise entails the hypothesis by syntax, shared words, or ordinary composition alone.
- Treat uncertainty as a reason to output fewer relations, not broader relations.
""",
)


CANDIDATE_PROMPTS = [
    CandidatePrompt(
        name="strict_bridge",
        description="Strict minimal KB bridge prompt with current isa_wn/disj schema.",
        template="""You generate knowledge-base relations for one Natural Language Inference pair.

Return only relations that can help a symbolic prover connect the Premise to the Hypothesis.

Allowed predicates:
- isa_wn(X, Y): X is a kind of Y, entails Y, or is a direct paraphrase useful for proving the hypothesis.
- disj(X, Y): X and Y are clear opposites or mutually exclusive.

Rules:
- Use only short phrases from, or very close to, the Premise and Hypothesis.
- Use lemmas for single words.
- Max 3 words per argument.
- Do not output modifier-dropping facts such as isa_wn(black dog, dog).
- Do not output morphology-only facts such as isa_wn(running, run).
- If no non-trivial bridge is needed, output NO_RELATION.
- Output no explanation.

Premise: ${premise}
Hypothesis: ${hypothesis}

Answer using exactly one of these formats:
NO_RELATION
or
[KB_START]
predicate(arg1, arg2)
predicate(arg1, arg2)
[KB_END]
""",
    ),
    CandidatePrompt(
        name="lasha_compact",
        description="Compact lexical-entailment prompt derived from Lasha's style.",
        template="""You are extracting lexical entailment relations for an NLI example.

Decide which short lexical bridges are needed for the Premise to entail the Hypothesis.
Output only needed relations. If the pair is not entailment, or no lexical bridge is needed, output NO_RELATION.

Use:
- entails(X, Y) when X from the premise semantically entails or normalizes to Y from the hypothesis.
- disj(X, Y) only for clear contradictions such as open/closed, empty/full, alive/dead.

Formatting constraints:
- No determiners.
- Lemmatize nouns and verbs.
- No auxiliary verbs.
- No prepositional phrases unless the whole phrase is essential.
- Avoid generic facts that are true but not needed for this NLI pair.

Examples:
Premise: A female swimmer is getting out of the pool.
Hypothesis: A woman gets out of the pool.
relations: { entails(female swimmer, woman) }

Premise: A tall man with a cap is climbing a cord.
Hypothesis: A man in a hat is climbing a rope.
relations: { entails(cap, hat), entails(cord, rope) }

Premise: A woman in red clothing is dancing inside a crowd.
Hypothesis: A woman in red is dancing in a crowd.
relations: { }

Now process:
premise: ${premise}
hypothesis: ${hypothesis}

Output exactly:
relations: { ... }
or
NO_RELATION
""",
    ),
    CandidatePrompt(
        name="precision_guarded",
        description="High-precision variant: only relation if clearly necessary.",
        template="""You are a conservative KB-injection annotator.

Your priority is precision: a wrong or speculative relation is worse than missing a weak relation.

Generate a relation only when all are true:
1. It bridges a concrete mismatch between Premise and Hypothesis.
2. It is directionally correct for proving the hypothesis.
3. It is not just deleting an adjective, determiner, tense marker, or plural.
4. It uses only isa_wn or disj.

Use isa_wn(specific, general_or_paraphrase) for lexical entailment.
Use disj(term1, term2) only for obvious incompatibility.

If uncertain, output NO_RELATION.

Premise: ${premise}
Hypothesis: ${hypothesis}

Final answer only:
[KB_START]
...
[KB_END]

If empty:
[KB_START]
[KB_END]
""",
    ),
    CandidatePrompt(
        name="recall_balanced",
        description="Balanced recall prompt: find all useful bridges but avoid trivial ones.",
        template="""Find all useful semantic KB relations that would help prove or refute the hypothesis from the premise.

Use only:
isa_wn(X, Y) for lexical entailment, type-of, paraphrase, or useful normalization.
disj(X, Y) for direct incompatibility.

Include every non-trivial relation that is likely useful:
- person/object/category bridges, e.g. guitarist -> person, drum -> musical instrument
- synonym/paraphrase bridges, e.g. cap -> hat, sofa -> couch
- event bridges, e.g. strum -> play, bounce -> jump
- location/generalization bridges, e.g. beach -> outdoors, sidewalk -> outside

Exclude:
- modifier dropping, e.g. red shirt -> shirt
- same lemma/inflection only
- generic background facts with no proof role
- uncertain disjunctions

Keep arguments short and lemmatized. Output only KB lines or NO_RELATION.

Premise: ${premise}
Hypothesis: ${hypothesis}

[KB_START]
""",
    ),
    CandidatePrompt(
        name="json_schema",
        description="JSON-shaped output to test whether stricter structure improves parseability.",
        template="""Extract semantic relations for this NLI pair.

Return valid JSON only:
{"relations": ["isa_wn(arg1, arg2)", "disj(arg1, arg2)"]}

Use an empty list when no relation is needed:
{"relations": []}

Relation policy:
- isa_wn(X, Y): X semantically entails, is a type of, or is a direct paraphrase of Y.
- disj(X, Y): X and Y are mutually exclusive.
- Prefer relations that bridge Premise terms to Hypothesis terms.
- Lemmatize short arguments.
- Do not include trivial modifier deletion or morphology-only relations.
- Do not include explanation text outside JSON.

Premise: ${premise}
Hypothesis: ${hypothesis}
""",
    ),
    CandidatePrompt(
        name="rewrite_minimal_lemma",
        description="Rewritten prompt: minimal lemma/content-word pairs, no whole-event bridges.",
        template="""You generate KB relations for an NLI prover.

Output only minimal lexical-semantic bridges between content words or short noun phrases.

Allowed:
- isa_wn(X, Y): X is a synonym, paraphrase, type, role, or category that directly supports Y.
- disj(X, Y): X and Y are direct opposites or mutually exclusive.

Very important:
- Use the smallest useful argument: prefer one content word over a clause.
- Use base forms for verbs and nouns when that is natural: slice -> cut, comb -> style.
- Do not output both an inflected form and its lemma.
- Do not bridge whole events or social implications, e.g. carrying a shopping bag does not imply shopping.
- Do not bridge modifier deletion, e.g. old woman -> woman, military men -> men, dog with big ears -> dog.
- Do not bridge syntactic paraphrases that need no lexical relation.
- If the only difference is tense, number, auxiliary verbs, or word order, output NO_RELATION.

Premise: ${premise}
Hypothesis: ${hypothesis}

Output exactly:
[KB_START]
relation lines only
[KB_END]

Use an empty block for NO_RELATION.
""",
    ),
    CandidatePrompt(
        name="rewrite_annotation_style",
        description="Rewritten prompt: mimic human annotation style without dev-item examples.",
        template="""Your task is to produce the kind of short KB pairs a human annotator would add for an NLI item.

Human annotation style:
- Usually 0, 1, or 2 relations.
- Relations are short pairs such as cap -> hat, child -> kid, lamb -> animal, outdoors -> outside.
- A relation should connect a real lexical mismatch between the premise and hypothesis.
- Keep only pairs that would still look correct if written as (X, Y).

Use isa_wn(X, Y) for the pair. Use disj(X, Y) only for genuine opposites.

Reject these:
- dropping modifiers: black dog -> dog, old woman -> woman
- inflection only: plays -> playing, dogs -> dog
- loose event implications: surf -> catch, carry bag -> shop
- whole clauses or phrases longer than 3 words
- background facts that are true but not needed for this sentence pair

If no human-style KB pair is needed, output NO_RELATION.

Premise: ${premise}
Hypothesis: ${hypothesis}

Answer only with:
NO_RELATION
or
isa_wn(x, y)
isa_wn(x, y)
""",
    ),
    CandidatePrompt(
        name="rewrite_category_guarded",
        description="Rewritten prompt: allow only high-confidence lexical categories.",
        template="""Extract only high-confidence lexical KB relations for this Premise/Hypothesis pair.

A relation is allowed only if it belongs to one of these categories:
1. synonym or near-synonym: cap -> hat, sofa -> couch
2. person/role/category: boy -> child, lamb -> animal
3. object category: drum -> musical instrument
4. adjective/state paraphrase: gleaming -> bright
5. location paraphrase: outdoors -> outside
6. clear opposition: open <-> closed, empty <-> full

Do not output:
- activity guesses or event consequences unless they are direct verb synonyms
- modifier removal
- same-lemma morphology
- long phrase-to-long phrase relations
- relations that merely restate that the hypothesis is entailed

Direction: from the premise wording to the hypothesis wording.

Premise: ${premise}
Hypothesis: ${hypothesis}

Final KB:
[KB_START]
predicate(arg1, arg2)
[KB_END]

If none:
[KB_START]
[KB_END]
""",
    ),
    CandidatePrompt(
        name="rewrite_two_pass_hidden",
        description="Rewritten prompt: hidden decision pass, final minimal KB only.",
        template="""First, silently compare the premise and hypothesis.
Second, silently decide whether a symbolic prover needs a lexical bridge.
Third, output only the final KB.

Use isa_wn(X, Y) only when X from the premise is a direct lexical bridge to Y from the hypothesis.
Use disj(X, Y) only for clear contradiction.

Selection tests for every relation:
- Would a human annotator likely write this pair as a short KB annotation?
- Is X a word/short phrase actually in the premise?
- Is Y a word/short phrase actually in the hypothesis?
- Is this more than morphology, modifier dropping, or syntax?
- Is this not a social/activity inference?

If any answer is no, omit the relation.

Premise: ${premise}
Hypothesis: ${hypothesis}

Return only:
[KB_START]
...
[KB_END]
""",
    ),
    CandidatePrompt(
        name="rewrite_json_minimal",
        description="Rewritten JSON prompt: strict minimal human-style relations.",
        template="""Return JSON only, with this schema:
{"relations": ["isa_wn(x, y)", "disj(x, y)"]}

Use [] if no relation is needed.

Guidelines:
- Prefer zero relations unless a short lexical bridge is clearly needed.
- Keep arguments minimal, usually one word or a compact noun phrase.
- Use premise wording on the left and hypothesis wording on the right.
- Use lemmas for verbs when possible.
- Never include both an inflected relation and a lemma relation.
- Never include modifier-dropping facts.
- Never include loose event/social implications.
- Never include explanations.

Premise: ${premise}
Hypothesis: ${hypothesis}
""",
    ),
    CandidatePrompt(
        name="ettore_plus_precision",
        description="Anchored Ettore prompt plus precision calibration.",
        template=ETTORE_PLUS_PRECISION,
    ),
    CandidatePrompt(
        name="ettore_plus_precision_soft",
        description="Anchored Ettore prompt plus softer precision and recovery calibration.",
        template=ETTORE_PLUS_PRECISION_SOFT,
    ),
    CandidatePrompt(
        name="ettore_plus_lemma_canonical",
        description="Anchored Ettore prompt plus canonical form calibration.",
        template=ETTORE_PLUS_LEMMA_CANONICAL,
    ),
    CandidatePrompt(
        name="ettore_plus_human_annotation",
        description="Anchored Ettore prompt plus human-annotation-style calibration.",
        template=ETTORE_PLUS_HUMAN_ANNOTATION,
    ),
    CandidatePrompt(
        name="lasha_plus_precision",
        description="Anchored Lasha prompt plus precision calibration.",
        template=LASHA_PLUS_PRECISION,
    ),
    CandidatePrompt(
        name="lasha_plus_precision_recovery",
        description="Anchored Lasha prompt plus precision and valid-event recovery calibration.",
        template=LASHA_PLUS_PRECISION_RECOVERY,
    ),
    CandidatePrompt(
        name="lasha_plus_minimal_schema",
        description="Anchored Lasha prompt plus minimal-schema calibration.",
        template=LASHA_PLUS_MINIMAL_SCHEMA,
    ),
]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip()).strip("_")


def generated_columns(prompt_name: str, model: str) -> GeneratedColumns:
    prefix = f"LLM__{prompt_name}__{model_slug(model)}"
    return GeneratedColumns(
        raw_response=f"{prefix}_raw_response",
        kb=f"{prefix}_KB",
        error=f"{prefix}_error",
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return reader.fieldnames, list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_columns(fieldnames: list[str], columns: Iterable[str]) -> list[str]:
    updated = list(fieldnames)
    for column in columns:
        if column not in updated:
            updated.append(column)
    return updated


def fill_template(template: str, row: dict[str, str]) -> str:
    premise = str(row.get("premise", "")).strip()
    hypothesis = str(row.get("hypothesis", "")).strip()
    return (
        template.replace("${premise}", premise)
        .replace("${hypothesis}", hypothesis)
        .replace("${PREMISE}", premise)
        .replace("${HYPOTHESIS}", hypothesis)
    )


def row_to_problem(row: dict[str, str]) -> NLIProblem:
    return NLIProblem(
        id=str(row.get("ID", "")).strip(),
        premises=[str(row.get("premise", "")).strip()],
        hypothesis=str(row.get("hypothesis", "")).strip(),
        gold_label=NLILabel(row.get("gold_label", "-") or "-"),
        dataset=str(row.get("dataset", "")).strip(),
        split=str(row.get("split", "")).strip(),
        original_data=dict(row),
    )


def format_kb(relations: list[str]) -> str:
    return "; ".join(relations) if relations else "NO_RELATION"


def normalize_relations(relations: list[str], problem: NLIProblem, *, filter_kb: bool) -> list[str]:
    if not filter_kb:
        return relations
    filtered = pipeline_filter_kb_injections(
        relations,
        problem.premises,
        problem.hypothesis,
        post_process=True,
    )
    return [result.relation for result in filtered]


def row_has_relation(row: dict[str, str], reference_columns: list[str]) -> bool:
    for column in reference_columns:
        value = row.get(column, "")
        if str(value or "").strip() and parse_kb_cell(value):
            return True
    return False


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[0]]
    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    selected = []
    for index in indexes:
        if index not in seen:
            selected.append(rows[index])
            seen.add(index)
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        if cursor not in seen:
            selected.append(rows[cursor])
            seen.add(cursor)
        cursor += 1
    return selected


def build_agreed_subset(
    agreement_csv: Path,
    agreed_subset_csv: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = read_rows(agreement_csv)
    subset = [
        row
        for row in rows
        if row.get("ID", "").strip()
        and row.get("all_present_exact_match", "").strip().upper() == "TRUE"
    ]
    write_rows(agreed_subset_csv, fieldnames, subset)
    return fieldnames, subset


def choose_dev_sample(
    rows: list[dict[str, str]],
    reference_columns: list[str],
    sample_size: int,
    relation_fraction: float,
) -> list[dict[str, str]]:
    relation_rows = [row for row in rows if row_has_relation(row, reference_columns)]
    no_relation_rows = [row for row in rows if not row_has_relation(row, reference_columns)]

    relation_target = min(len(relation_rows), math.ceil(sample_size * relation_fraction))
    no_relation_target = min(len(no_relation_rows), sample_size - relation_target)
    if relation_target + no_relation_target < sample_size:
        relation_target = min(len(relation_rows), relation_target + sample_size - relation_target - no_relation_target)
    if relation_target + no_relation_target < sample_size:
        no_relation_target = min(len(no_relation_rows), sample_size - relation_target)

    selected = evenly_spaced(relation_rows, relation_target) + evenly_spaced(
        no_relation_rows,
        no_relation_target,
    )
    selected_ids = {row.get("ID", "") for row in selected}
    remaining = [row for row in rows if row.get("ID", "") not in selected_ids]
    selected.extend(evenly_spaced(remaining, sample_size - len(selected)))
    order = {row.get("ID", ""): index for index, row in enumerate(rows)}
    return sorted(selected[:sample_size], key=lambda row: order.get(row.get("ID", ""), 10**9))


def copy_baseline_rows(
    baseline_csv: Path,
    sample_rows: list[dict[str, str]],
    baseline_sample_csv: Path,
    *,
    exclude_model_fragments: list[str],
) -> Path | None:
    if not baseline_csv.exists():
        return None
    fieldnames, rows = read_rows(baseline_csv)
    sample_ids = {row["ID"] for row in sample_rows}
    filtered_rows = [row for row in rows if row.get("ID", "") in sample_ids]
    allowed_fieldnames = [
        name
        for name in fieldnames
        if not any(fragment in name for fragment in exclude_model_fragments)
    ]
    trimmed_rows = [
        {field: row.get(field, "") for field in allowed_fieldnames}
        for row in filtered_rows
    ]
    write_rows(baseline_sample_csv, allowed_fieldnames, trimmed_rows)
    return baseline_sample_csv


async def call_with_retries(
    client: AsyncGenericAIClient,
    *,
    prompt: str,
    model: str,
    max_retries: int,
    request_timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            output = await asyncio.wait_for(
                client.generate(prompt=prompt, model=model),
                timeout=request_timeout,
            )
            text = str(output or "").strip()
            if not text:
                raise RuntimeError("empty model response")
            return text
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(str(last_error))


async def run_generation(args: argparse.Namespace, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output_path = Path(args.output_csv)
    if args.resume and output_path.exists():
        fieldnames, rows = read_rows(output_path)

    prompts = [prompt for prompt in CANDIDATE_PROMPTS if prompt.name in args.prompts]
    prompt_names = {prompt.name for prompt in prompts}
    missing = [name for name in args.prompts if name not in prompt_names]
    if missing:
        raise SystemExit(f"Unknown candidate prompt(s): {', '.join(missing)}")

    generated = []
    for prompt in prompts:
        for model in args.models:
            columns = generated_columns(prompt.name, model)
            generated.extend([columns.raw_response, columns.kb, columns.error])
    fieldnames = ensure_columns(fieldnames, generated)
    for row in rows:
        for column in generated:
            row.setdefault(column, "")

    client = AsyncGenericAIClient(provider=args.provider)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    progress = {"done": 0}

    jobs = []
    for row in rows:
        for prompt in prompts:
            for model in args.models:
                columns = generated_columns(prompt.name, model)
                if args.resume and (row.get(columns.kb, "").strip() or row.get(columns.error, "").strip()):
                    progress["done"] += 1
                    continue
                jobs.append((row, prompt, model, columns))

    total = len(rows) * len(prompts) * len(args.models)
    if progress["done"]:
        print(f"Resume skipped {progress['done']}/{total} already-filled jobs.", flush=True)
    print(f"Running {len(jobs)} jobs; total grid size is {total}.", flush=True)

    async def run_one(job: tuple[dict[str, str], CandidatePrompt, str, GeneratedColumns]) -> None:
        row, prompt, model, columns = job
        async with semaphore:
            row[columns.raw_response] = ""
            row[columns.kb] = ""
            row[columns.error] = ""
            try:
                filled = fill_template(prompt.template, row)
                raw = await call_with_retries(
                    client,
                    prompt=filled,
                    model=model,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout,
                )
                row[columns.raw_response] = raw
                parsed = _extract_validated_kb_from_output(raw)
                normalized = normalize_relations(
                    parsed,
                    row_to_problem(row),
                    filter_kb=not args.no_filter_kb,
                )
                row[columns.kb] = format_kb(normalized)
            except Exception as exc:
                row[columns.error] = str(exc)
            async with write_lock:
                progress["done"] += 1
                if progress["done"] % args.write_every_jobs == 0 or progress["done"] == total:
                    write_rows(output_path, fieldnames, rows)
                    print(f"Wrote {output_path} after {progress['done']}/{total} jobs.", flush=True)

    if jobs:
        await asyncio.gather(*(run_one(job) for job in jobs))
    write_rows(output_path, fieldnames, rows)


def score_outputs(
    csv_path: Path,
    summary_csv: Path,
    reference_columns: list[str],
    *,
    empty_prediction_is_no_relation: bool,
    metadata_csv: Path | None = None,
) -> list[EvaluationResult]:
    fieldnames, rows = read_rows(csv_path)
    prediction_columns = [
        name for name in fieldnames if name.startswith("LLM__") and name.endswith("_KB")
    ]
    results = [
        evaluate_prediction_column(
            rows,
            prediction_column,
            reference_columns,
            empty_prediction_is_no_relation=empty_prediction_is_no_relation,
        )
        for prediction_column in prediction_columns
    ]
    write_summary(summary_csv, results)
    if metadata_csv is not None:
        write_score_metadata(metadata_csv, results)
    return results


def parse_prediction_column(column: str) -> tuple[str, str]:
    body = column.removeprefix("LLM__").removesuffix("_KB")
    if "__" not in body:
        return body, ""
    prompt, model = body.split("__", 1)
    return prompt, model


def write_score_metadata(path: Path, results: list[EvaluationResult]) -> None:
    fieldnames = [
        "rank",
        "prompt",
        "model",
        "prediction_column",
        "micro_f1",
        "precision",
        "recall",
        "exact_best_match_rate",
        "evaluated_items",
        "tp",
        "fp",
        "fn",
        "exact_best_matches",
        "no_relation_best_matches",
        "skipped_missing_prediction",
        "skipped_no_reference",
    ]
    sorted_results = sorted(
        results,
        key=lambda result: (
            -(-1.0 if math.isnan(result.selected_counts.f1) else result.selected_counts.f1),
            result.prediction_column,
        ),
    )
    rows = []
    for rank, result in enumerate(sorted_results, start=1):
        prompt, model = parse_prediction_column(result.prediction_column)
        counts = result.selected_counts
        exact_rate = result.exact_best_matches / result.evaluated_items if result.evaluated_items else float("nan")
        rows.append(
            {
                "rank": rank,
                "prompt": prompt,
                "model": model,
                "prediction_column": result.prediction_column,
                "micro_f1": format_score(counts.f1),
                "precision": format_score(counts.precision),
                "recall": format_score(counts.recall),
                "exact_best_match_rate": format_score(exact_rate),
                "evaluated_items": result.evaluated_items,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "exact_best_matches": result.exact_best_matches,
                "no_relation_best_matches": result.no_relation_best_matches,
                "skipped_missing_prediction": result.skipped_missing_prediction,
                "skipped_no_reference": result.skipped_no_reference,
            }
        )
    write_rows(path, fieldnames, rows)


def write_candidate_prompt_docs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Candidate Prompts", ""]
    for prompt in CANDIDATE_PROMPTS:
        lines.extend(
            [
                f"## {prompt.name}",
                "",
                prompt.description,
                "",
                "```text",
                prompt.template,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agreement-csv",
        default=str(
            ROOT / "data" / "annotator_agreement" / "iaa_overview_edit.csv"
        ),
        help="Edited IAA overview used to derive the high-agreement subset.",
    )
    parser.add_argument(
        "--agreed-subset-csv",
        help=(
            "Optional destination for the derived high-agreement subset. "
            "Defaults beside --sample-csv."
        ),
    )
    parser.add_argument(
        "--sample-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_sample.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_outputs.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_summary.csv"),
    )
    parser.add_argument(
        "--leaderboard-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_leaderboard.csv"),
    )
    parser.add_argument(
        "--baseline-csv",
        default=str(ROOT / "llm_multi_reference_outputs_small_models.csv"),
    )
    parser.add_argument(
        "--baseline-sample-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_baseline_outputs.csv"),
    )
    parser.add_argument(
        "--baseline-leaderboard-csv",
        default=str(ROOT / "prompt_engineering" / "prompt_trial_30_baseline_leaderboard.csv"),
    )
    parser.add_argument(
        "--candidate-doc",
        default=str(ROOT / "prompt_engineering" / "candidate_prompts.md"),
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument(
        "--relation-fraction",
        type=float,
        default=2 / 3,
        help="Target fraction of sampled rows containing at least one reference relation.",
    )
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompts", nargs="+", default=[prompt.name for prompt in CANDIDATE_PROMPTS])
    parser.add_argument("--reference-columns", nargs="+", default=DEFAULT_REFERENCE_COLUMNS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--write-every-jobs", type=int, default=20)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-filter-kb", action="store_true")
    parser.add_argument("--empty-prediction-is-no-relation", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create agreed subset, dev sample, prompt docs, and baseline sample without API calls.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    agreement_csv = Path(args.agreement_csv)
    sample_csv = Path(args.sample_csv)
    agreed_subset_csv = (
        Path(args.agreed_subset_csv)
        if args.agreed_subset_csv
        else sample_csv.with_name("agreed_subset.csv")
    )
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    leaderboard_csv = Path(args.leaderboard_csv)

    fieldnames, subset = build_agreed_subset(agreement_csv, agreed_subset_csv)
    sample_rows = choose_dev_sample(
        subset,
        args.reference_columns,
        args.sample_size,
        args.relation_fraction,
    )
    write_rows(sample_csv, fieldnames, sample_rows)
    write_candidate_prompt_docs(Path(args.candidate_doc))

    baseline_sample = copy_baseline_rows(
        Path(args.baseline_csv),
        sample_rows,
        Path(args.baseline_sample_csv),
        exclude_model_fragments=["gemma"],
    )
    if baseline_sample is not None:
        score_outputs(
            baseline_sample,
            Path(args.baseline_sample_csv).with_name("prompt_trial_30_baseline_summary.csv"),
            args.reference_columns,
            empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
            metadata_csv=Path(args.baseline_leaderboard_csv),
        )

    if args.prepare_only:
        print(f"Wrote agreed subset: {agreed_subset_csv} ({len(subset)} rows)")
        print(f"Wrote dev sample: {sample_csv} ({len(sample_rows)} rows)")
        print("Prepare-only mode: no LLM calls made.")
        return

    await run_generation(args, sample_rows, fieldnames)
    score_outputs(
        output_csv,
        summary_csv,
        args.reference_columns,
        empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
        metadata_csv=leaderboard_csv,
    )
    print(f"Wrote agreed subset: {agreed_subset_csv} ({len(subset)} rows)")
    print(f"Wrote dev sample: {sample_csv} ({len(sample_rows)} rows)")
    print(f"Wrote outputs: {output_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote leaderboard: {leaderboard_csv}")


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1.")
    if not 0 <= args.relation_fraction <= 1:
        raise SystemExit("--relation-fraction must be between 0 and 1.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1.")
    if args.write_every_jobs < 1:
        raise SystemExit("--write-every-jobs must be >= 1.")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be greater than 0.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
