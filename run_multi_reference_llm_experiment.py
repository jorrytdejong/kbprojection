#!/usr/bin/env python3
"""Generate LLM KB columns for multi-reference human annotation evaluation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from kbprojection.filtering import pipeline_filter_kb_injections
from kbprojection.llm import (
    AsyncGenericAIClient,
    _extract_validated_kb_from_output,
)
from kbprojection.models import NLILabel, NLIProblem
from kbprojection.prompts import fill_prompt, list_prompts


DEFAULT_INPUT_CSV = "annotator agreement - inter_annotator_agreement_overview.csv"
DEFAULT_OUTPUT_CSV = "llm_multi_reference_outputs.csv"
DEFAULT_PROMPTS = ["ettore", "lasha"]
REPOSITORY_ROOT = Path(__file__).resolve().parent
CANONICAL_INPUT_CSV = REPOSITORY_ROOT / "data" / "all_usable_items_362.csv"
EXPECTED_INPUT_ROWS = 362
REQUIRED_INPUT_COLUMNS = {
    "ID",
    "premise",
    "hypothesis",
    "gold_label",
    "Alternative_KB",
    "Ettore_KB",
    "Jorryt_KB",
    "Lasha_KB",
    "Stefan_KB",
}


@dataclass(frozen=True)
class GeneratedColumns:
    raw_response: str
    kb: str
    error: str


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip()).strip("_")


def generated_columns(prompt: str, model: str) -> GeneratedColumns:
    prefix = f"LLM__{prompt}__{model_slug(model)}"
    return GeneratedColumns(
        raw_response=f"{prefix}_raw_response",
        kb=f"{prefix}_KB",
        error=f"{prefix}_error",
    )


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return reader.fieldnames, list(reader)


def validate_input_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}"
        )
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_INPUT_ROWS} data rows; "
            f"found {len(rows)}"
        )
    if any(not row.get("ID", "").strip() for row in rows):
        raise ValueError(f"{path} contains a blank ID value.")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
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
    if not relations:
        return "NO_RELATION"
    return "; ".join(relations)


def normalize_kb(
    relations: list[str],
    problem: NLIProblem,
    *,
    filter_kb: bool,
) -> list[str]:
    if not filter_kb:
        return relations
    filtered = pipeline_filter_kb_injections(
        relations,
        problem.premises,
        problem.hypothesis,
        post_process=True,
    )
    return [result.relation for result in filtered]


async def generate_raw_response(
    client: AsyncGenericAIClient,
    *,
    prompt: str,
    model: str,
    problem: NLIProblem,
    mock_response: str | None,
) -> str:
    filled_prompt = fill_prompt(prompt, problem.premises, problem.hypothesis)
    if mock_response is not None:
        return mock_response
    output = await client.generate(prompt=filled_prompt, model=model)
    return str(output or "")


def should_skip(row: dict[str, str], columns: GeneratedColumns, *, resume: bool) -> bool:
    if not resume:
        return False
    return bool(str(row.get(columns.kb, "")).strip() or str(row.get(columns.error, "")).strip())


async def run_experiment(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    input_fieldnames, input_rows = load_rows(input_path)
    validate_input_rows(input_path, input_fieldnames, input_rows)
    source_path = output_path if args.resume and output_path.exists() else input_path
    fieldnames, rows = load_rows(source_path)
    rows = rows[: args.limit] if args.limit is not None else rows

    available_prompts = set(list_prompts())
    missing_prompts = [prompt for prompt in args.prompts if prompt not in available_prompts]
    if missing_prompts:
        raise SystemExit(
            f"Unknown prompt(s): {', '.join(missing_prompts)}. "
            f"Available: {', '.join(sorted(available_prompts))}"
        )

    all_generated_columns: list[str] = []
    for prompt in args.prompts:
        for model in args.models:
            columns = generated_columns(prompt, model)
            all_generated_columns.extend([columns.raw_response, columns.kb, columns.error])
    fieldnames = ensure_columns(fieldnames, all_generated_columns)
    for row in rows:
        for column in all_generated_columns:
            row.setdefault(column, "")

    client = None if args.mock_response is not None else AsyncGenericAIClient(provider=args.provider)

    total_jobs = len(rows) * len(args.prompts) * len(args.models)
    completed_jobs = 0
    for row_index, row in enumerate(rows, start=1):
        problem = row_to_problem(row)
        for prompt in args.prompts:
            for model in args.models:
                columns = generated_columns(prompt, model)
                if should_skip(row, columns, resume=args.resume):
                    completed_jobs += 1
                    continue

                row[columns.raw_response] = ""
                row[columns.kb] = ""
                row[columns.error] = ""
                try:
                    raw_response = await generate_raw_response(
                        client,  # type: ignore[arg-type]
                        prompt=prompt,
                        model=model,
                        problem=problem,
                        mock_response=args.mock_response,
                    )
                    row[columns.raw_response] = raw_response
                    parsed_relations = _extract_validated_kb_from_output(raw_response)
                    normalized_relations = normalize_kb(
                        parsed_relations,
                        problem,
                        filter_kb=not args.no_filter_kb,
                    )
                    row[columns.kb] = format_kb(normalized_relations)
                except Exception as exc:
                    row[columns.error] = str(exc)
                completed_jobs += 1

        if row_index % args.write_every == 0 or row_index == len(rows):
            write_rows(output_path, fieldnames, rows)
            print(f"Wrote {output_path} after row {row_index}/{len(rows)} ({completed_jobs}/{total_jobs} jobs).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LLM KB columns for multi-reference evaluation."
    )
    parser.add_argument("--input-csv", default=str(CANONICAL_INPUT_CSV))
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip generated cells that already have a KB value or error.",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=1,
        help="Write the output CSV after this many rows.",
    )
    parser.add_argument(
        "--no-filter-kb",
        action="store_true",
        help="Parse relations but skip kbprojection's premise/hypothesis filtering.",
    )
    parser.add_argument(
        "--mock-response",
        help="Use this fixed model response instead of calling the provider; intended for smoke tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.write_every < 1:
        raise SystemExit("--write-every must be >= 1.")
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
