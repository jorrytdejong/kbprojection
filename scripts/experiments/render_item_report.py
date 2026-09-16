#!/usr/bin/env python3
"""Render a no-API Markdown inspection report for one saved LEX prediction.

The report joins a row from the curated multi-reference dataset with one saved
model output.  It never calls an LLM or LangPro; it only reads committed CSVs
and reuses the project's directed relation scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from calculate_multi_reference_f1 import (
    argument_order_agnostic_relation_counts,
    item_selection_score,
    model_prediction_exclusion,
    parse_kb_cell,
    relation_counts,
)


DEFAULT_ITEMS_CSV = REPOSITORY_ROOT / "data" / "all_usable_items_362.csv"
REFERENCE_COLUMNS = (
    "Alternative_KB",
    "Ettore_KB",
    "Jorryt_KB",
    "Lasha_KB",
    "Stefan_KB",
)
CANONICAL_PROMPT_NAME = "Lasha Plus Precision"
INTRINSIC_LEX_LABELS = frozenset({"lasha", "lasha_plus_precision"})
AGENTIC_LEX_LABELS = frozenset({"lex", "lasha_plus_precision"})


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return list(reader.fieldnames), list(reader)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def format_relation(relation: tuple[str, ...]) -> str:
    if len(relation) == 2:
        return f"{relation[0]} ⊑ {relation[1]}"
    return "(" + ", ".join(relation) + ")"


def format_relations(relations: frozenset[tuple[str, ...]]) -> list[str]:
    if not relations:
        return ["No lexical relation is needed."]
    return [format_relation(relation) for relation in sorted(relations)]


def format_number(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def item_f1(tp: int, fp: int, fn: int) -> float:
    """Return an intuitive perfect score when both relation sets are empty."""
    if tp == fp == fn == 0:
        return 1.0
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else float("nan")


def markdown_code_block(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text.rstrip()}\n{fence}"


def select_prediction(
    rows: list[dict[str, str]], *, item_id: str, prompt: str, model: str, repeat: int
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("ID") == item_id
        and row.get("prompt") == prompt
        and row.get("model") == model
        and row.get("repeat") == str(repeat)
    ]
    if not matches:
        raise ValueError(
            "No saved prediction matches "
            f"ID={item_id!r}, prompt={prompt!r}, model={model!r}, repeat={repeat}."
        )
    if len(matches) > 1:
        raise ValueError("More than one saved prediction matches the requested slot.")
    return matches[0]


def validate_lex_sources(prediction_row: dict[str, str], langpro_config: dict | None) -> None:
    """Keep reports within one conceptual prompt lineage despite legacy labels."""
    intrinsic_label = prediction_row.get("prompt", "")
    if intrinsic_label not in INTRINSIC_LEX_LABELS:
        raise ValueError(
            "This report is limited to Lasha Plus Precision saved outputs; "
            f"got intrinsic prompt label {intrinsic_label!r}."
        )
    if langpro_config is not None:
        agentic_label = str(langpro_config.get("prompt_arm", ""))
        if agentic_label not in AGENTIC_LEX_LABELS:
            raise ValueError(
                "This report is limited to the Lex agentic form of Lasha Plus Precision; "
                f"got saved LangPro profile label {agentic_label!r}."
            )


def load_langpro_record(path: Path, item_id: str) -> tuple[dict, dict]:
    """Load one durable agentic LangPro record and its run configuration."""
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    records = [record for record in artifact.get("records", []) if record.get("problem_id") == item_id]
    if not records:
        raise ValueError(f"No LangPro record for ID {item_id!r} in {path}.")
    if len(records) > 1:
        raise ValueError(f"More than one LangPro record for ID {item_id!r} in {path}.")
    return dict(artifact.get("config", {})), records[0]


def bullet_list(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values] if values else ["- None"]


def render_langpro_section(config: dict, record: dict, artifact_path: Path) -> list[str]:
    """Render one saved generate-prove-refine timeline without invoking LangPro."""
    outcome = record.get("outcome", {})
    attempts: dict[int, dict[str, dict]] = {}
    baseline: dict | None = None
    for event in record.get("timeline", []):
        phase = event.get("phase")
        attempt = event.get("attempt")
        if phase == "baseline_langpro":
            baseline = event
        elif isinstance(attempt, int):
            attempts.setdefault(attempt, {})[phase] = event

    lines = [
        "",
        "## Saved LangPro proof run",
        "",
        "This section replays a saved generate-prove-refine timeline; it does not call LangPro.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run | {config.get('run', '')} |",
        f"| Model | {config.get('model', '')} |",
        f"| Prompt | {CANONICAL_PROMPT_NAME} (agentic KB format) |",
        f"| Built-in WordNet | {config.get('langpro_builtin', '')} |",
        f"| Baseline prediction | {outcome.get('pred_baseline', '')} |",
        f"| Final prediction | {outcome.get('pred_final', '')} |",
        f"| Entailment proved | {'Yes' if outcome.get('solved') else 'No'} |",
        f"| Stop reason | {outcome.get('stop_reason', '')} |",
    ]
    if baseline:
        baseline_output = baseline.get("output", {})
        lines.extend([
            "",
            "### Baseline: LangPro with its default lexical knowledge",
            "",
            f"- Prediction: **{baseline_output.get('pred', 'unknown')}**",
            f"- Prover error: {baseline_output.get('prover_error') or 'None'}",
        ])

    for attempt_number in sorted(attempts):
        events = attempts[attempt_number]
        generated = events.get("kb_generation", {}).get("output", {})
        filtered = events.get("kb_filtering", {}).get("output", {})
        proved = events.get("langpro_with_kb", {}).get("output", {})
        critic = events.get("critic", {}).get("output", {})
        proof_excerpts = proved.get("proof_excerpts", {})
        entailment_closed = proof_excerpts.get("entailment", {}).get("closed")
        contradiction_closed = proof_excerpts.get("contradiction", {}).get("closed")
        lines.extend([
            "",
            f"### Attempt {attempt_number}",
            "",
            "**Model proposed**",
            *bullet_list(list(generated.get("kb_raw", []))),
            "",
            "**Relations that reached LangPro after filtering**",
            *bullet_list(list(filtered.get("kb_filtered", []))),
            "",
            "**LangPro result**",
            f"- Prediction: **{proved.get('pred', 'not run')}**",
            f"- Entailment branch closed: {entailment_closed if entailment_closed is not None else 'not recorded'}",
            f"- Contradiction branch closed: {contradiction_closed if contradiction_closed is not None else 'not recorded'}",
            f"- All injected relations echoed by LangPro: {proved.get('kb_verification', {}).get('all_sent_found_in_response', 'not recorded')}",
        ])
        analysis_lines = list(critic.get("analysis_lines", []))
        if analysis_lines:
            lines.extend([
                "",
                "<details>",
                "<summary>Critic feedback</summary>",
                "",
                *analysis_lines,
                "",
                "</details>",
            ])

    lines.extend([
        "",
        f"- LangPro artifact source: `{display_path(artifact_path)}`",
    ])
    return lines


def render_report(
    item: dict[str, str],
    prediction_row: dict[str, str],
    *,
    items_csv: Path,
    predictions_csv: Path,
    langpro_config: dict | None = None,
    langpro_record: dict | None = None,
    langpro_results_json: Path | None = None,
) -> str:
    prediction = parse_kb_cell(prediction_row.get("KB", ""))
    exclusion = model_prediction_exclusion(prediction_row, "KB")
    raw_references = [
        parse_kb_cell(item.get(column, ""))
        for column in REFERENCE_COLUMNS
        if item.get(column, "").strip()
    ]
    references = [
        (f"Human explanation {number}", reference)
        for number, reference in enumerate(raw_references, start=1)
    ]
    if not references:
        raise ValueError(f"Item {item.get('ID')!r} has no human LEX references.")

    lines = [
        f"# KB Projection item report: {item['ID']}",
        "",
        "## Input",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Dataset | {item.get('dataset', '')} / {item.get('split', '')} |",
        f"| Gold label | {item.get('gold_label', '')} |",
        f"| Premise | {item.get('premise', '')} |",
        f"| Hypothesis | {item.get('hypothesis', '')} |",
        "",
        "## Human lexical explanations",
        "",
        "Different annotators can provide different valid explanations for the same item.",
    ]
    for column, reference in references:
        lines.extend(["", f"### {column}", ""])
        lines.extend(f"- {relation}" for relation in format_relations(reference))

    lines.extend([
        "",
        "## Saved model output",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Prompt | {CANONICAL_PROMPT_NAME} |",
        f"| Model | {prediction_row.get('model', '')} |",
        f"| Repeat | {prediction_row.get('repeat', '')} |",
        f"| Saved error | {prediction_row.get('error', '').strip() or 'None'} |",
        "",
        "### Raw response",
        "",
        markdown_code_block(prediction_row.get("raw_response", "") or "(empty response)"),
        "",
        "### Parsed lexical relations",
        "",
    ])
    lines.extend(f"- {relation}" for relation in format_relations(prediction))

    lines.extend(["", "## Comparison with human explanations", ""])
    if exclusion == "error":
        lines.append("This output is not scored because the saved run has an API or parsing error.")
    elif exclusion == "non_entailment":
        lines.append("This output is not scored because its final explicit answer is `non-entailment`.")
    else:
        lines.extend([
            "Each row compares the model's relations with one valid human explanation.",
            "",
            "| Human explanation | Matched | Extra | Missing | Similarity |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        scored = []
        for column, reference in references:
            counts = relation_counts(prediction, reference)
            scored.append((item_selection_score(counts), column, counts))
            lines.append(
                f"| {column} | {counts.tp} | {counts.fp} | {counts.fn} | "
                f"{format_number(item_f1(counts.tp, counts.fp, counts.fn))} |"
            )
        score, selected_column, counts = max(
            scored, key=lambda entry: (entry[0], entry[2].tp, -entry[2].fp, -entry[2].fn, entry[1])
        )
        direction_free = argument_order_agnostic_relation_counts(
            prediction, dict(references)[selected_column]
        )
        lines.extend([
            "",
            f"Overall similarity: **{format_number(score)}**. For one-number scoring, the evaluator uses "
            "the closest valid human explanation; the table above shows every explanation.",
            f"Direction-insensitive diagnostic: **{format_number(item_f1(direction_free.tp, direction_free.fp, direction_free.fn))}**.",
            "",
            "### What to inspect next",
            "",
            "- **Matched** relations are shared by the model and this human explanation.",
            "- **Extra** relations were generated by the model but are absent from this explanation.",
            "- **Missing** relations occur in this explanation but not in the model output.",
        ])

    if langpro_config is not None and langpro_record is not None and langpro_results_json is not None:
        lines.extend(render_langpro_section(langpro_config, langpro_record, langpro_results_json))

    lines.extend([
        "",
        "## Report provenance",
        "",
        f"- Item source: `{display_path(items_csv)}`",
        f"- Saved output source: `{display_path(predictions_csv)}`",
        "- This report performed no model or LangPro calls.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--items-csv", type=Path, default=DEFAULT_ITEMS_CSV)
    parser.add_argument(
        "--langpro-results-json",
        type=Path,
        help="Optional saved agentic LangPro artifact to append as a proof-run section.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, items = read_csv(args.items_csv)
    _, predictions = read_csv(args.predictions_csv)
    item = next((row for row in items if row.get("ID") == args.item_id), None)
    if item is None:
        raise ValueError(f"No item with ID {args.item_id!r} in {args.items_csv}.")
    prediction = select_prediction(
        predictions,
        item_id=args.item_id,
        prompt=args.prompt,
        model=args.model,
        repeat=args.repeat,
    )
    if prediction.get("premise") != item.get("premise") or prediction.get("hypothesis") != item.get("hypothesis"):
        raise ValueError("Saved prediction text does not match the curated item source.")
    langpro_config = langpro_record = None
    if args.langpro_results_json:
        langpro_config, langpro_record = load_langpro_record(args.langpro_results_json, args.item_id)
        problem = langpro_record.get("problem", {})
        if "\n".join(problem.get("premise", [])) != item.get("premise") or problem.get("hypothesis") != item.get("hypothesis"):
            raise ValueError("Saved LangPro problem text does not match the curated item source.")
    validate_lex_sources(prediction, langpro_config)
    report = render_report(
        item,
        prediction,
        items_csv=args.items_csv,
        predictions_csv=args.predictions_csv,
        langpro_config=langpro_config,
        langpro_record=langpro_record,
        langpro_results_json=args.langpro_results_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
