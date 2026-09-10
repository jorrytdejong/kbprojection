#!/usr/bin/env python3
"""Run and evaluate repeated multi-reference KB generation experiments.

The output is long-format: one row per item/prompt/model/repeat. Metrics are
computed from parsed KB cells, so they reflect the same normalized KB comparison
used by the multi-reference scorer.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "experiment_results" / "lasha_all362_5runs"
DEFAULT_INPUT_CSV = ROOT / "data" / "all_usable_items_362.csv"
EXPECTED_INPUT_ROWS = 362

from calculate_multi_reference_f1 import (
    Counts,
    evaluate_prediction_column,
    parse_kb_cell,
    relation_counts,
)
from kbprojection.filtering import FILTERING_PIPELINE_VERSION, pipeline_filter_kb_injections
from kbprojection.llm import AsyncGenericAIClient, _extract_validated_kb_from_output
from kbprojection.models import FilteringConfig, NLILabel, NLIProblem
from kbprojection.prompts import fill_prompt


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


def validate_input_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    reference_columns: list[str],
) -> None:
    required = {
        "ID",
        "premise",
        "hypothesis",
        "gold_label",
        *reference_columns,
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}"
        )
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_INPUT_ROWS} data rows; "
            f"found {len(rows)}"
        )
    blank_ids = [index + 2 for index, row in enumerate(rows) if not row.get("ID", "").strip()]
    if blank_ids:
        raise ValueError(f"{path} contains blank ID values on CSV lines: {blank_ids[:10]}")
DEFAULT_PROMPTS = ["lasha", "ettore"]
OUTPUT_FIELDNAMES = [
    "ID",
    "dataset",
    "split",
    "premise",
    "hypothesis",
    "gold_label",
    "has_reference_relation",
    "prompt",
    "model",
    "repeat",
    "raw_response",
    "KB",
    "error",
    "filtering_config",
    "filtering_version",
]


@dataclass(frozen=True)
class JobKey:
    item_id: str
    prompt: str
    model: str
    repeat: int


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return reader.fieldnames, list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip()).strip("_")


def format_kb(relations: list[str]) -> str:
    return "; ".join(relations) if relations else "NO_RELATION"


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


def has_reference_relation(row: dict[str, str], reference_columns: list[str]) -> bool:
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
    selected = []
    seen: set[int] = set()
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


def choose_balanced_sample(
    rows: list[dict[str, str]],
    *,
    sample_size: int,
    reference_columns: list[str],
) -> list[dict[str, str]]:
    usable = [row for row in rows if row.get("ID", "").strip()]
    relation_rows = [row for row in usable if has_reference_relation(row, reference_columns)]
    no_relation_rows = [row for row in usable if not has_reference_relation(row, reference_columns)]

    relation_target = min(len(relation_rows), sample_size // 2)
    no_relation_target = min(len(no_relation_rows), sample_size - relation_target)
    if relation_target + no_relation_target < sample_size:
        relation_target = min(len(relation_rows), sample_size - no_relation_target)
    if relation_target + no_relation_target < sample_size:
        no_relation_target = min(len(no_relation_rows), sample_size - relation_target)

    selected = evenly_spaced(relation_rows, relation_target) + evenly_spaced(
        no_relation_rows,
        no_relation_target,
    )
    selected_ids = {row["ID"] for row in selected}
    remaining = [row for row in usable if row["ID"] not in selected_ids]
    selected.extend(evenly_spaced(remaining, sample_size - len(selected)))

    order = {row["ID"]: index for index, row in enumerate(rows)}
    return sorted(selected[:sample_size], key=lambda row: order.get(row["ID"], 10**9))


def normalize_relations(
    relations: list[str], problem: NLIProblem, *, filter_kb: bool,
    filtering: FilteringConfig | None = None,
) -> list[str]:
    if not filter_kb:
        return relations
    filtered = pipeline_filter_kb_injections(
        relations,
        problem.premises,
        problem.hypothesis,
        filtering=filtering or FilteringConfig.operational(),
    )
    return [result.relation for result in filtered]


def load_existing_outputs(path: Path) -> dict[JobKey, dict[str, str]]:
    if not path.exists():
        return {}
    _, rows = read_rows(path)
    existing: dict[JobKey, dict[str, str]] = {}
    for row in rows:
        try:
            repeat = int(row.get("repeat", ""))
        except ValueError:
            continue
        key = JobKey(
            item_id=row.get("ID", ""),
            prompt=row.get("prompt", ""),
            model=row.get("model", ""),
            repeat=repeat,
        )
        if key.item_id and key.prompt and key.model:
            existing[key] = row
    return existing


def build_output_rows(
    sample_rows: list[dict[str, str]],
    prompts: list[str],
    models: list[str],
    repeats: int,
    reference_columns: list[str],
    existing: dict[JobKey, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_row in sample_rows:
        item_id = source_row["ID"]
        for prompt in prompts:
            for model in models:
                for repeat in range(1, repeats + 1):
                    key = JobKey(item_id, prompt, model, repeat)
                    if key in existing:
                        rows.append(existing[key])
                        continue
                    rows.append(
                        {
                            "ID": item_id,
                            "dataset": source_row.get("dataset", ""),
                            "split": source_row.get("split", ""),
                            "premise": source_row.get("premise", ""),
                            "hypothesis": source_row.get("hypothesis", ""),
                            "gold_label": source_row.get("gold_label", ""),
                            "has_reference_relation": str(
                                has_reference_relation(source_row, reference_columns)
                            ).upper(),
                            "prompt": prompt,
                            "model": model,
                            "repeat": repeat,
                            "raw_response": "",
                            "KB": "",
                            "error": "",
                        }
                    )
    return rows


def is_done(row: dict[str, object]) -> bool:
    return bool(str(row.get("KB", "")).strip() or str(row.get("error", "")).strip())


async def call_with_retries(
    client: AsyncGenericAIClient,
    *,
    prompt_text: str,
    model: str,
    temperature: float | None,
    max_retries: int,
    request_timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            output = await asyncio.wait_for(
                client.generate(
                    prompt=prompt_text,
                    model=model,
                    temperature=temperature,
                ),
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


async def run_generation(
    rows: list[dict[str, object]],
    *,
    output_csv: Path,
    provider: str,
    concurrency: int,
    write_every_jobs: int,
    max_retries: int,
    request_timeout: float,
    temperature: float | None,
    filter_kb: bool,
    filtering: FilteringConfig | None = None,
) -> None:
    filtering = filtering or FilteringConfig.operational()
    client = AsyncGenericAIClient(provider=provider)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = sum(1 for row in rows if is_done(row))
    total = len(rows)
    jobs = [row for row in rows if not is_done(row)]

    if completed:
        print(f"Resume skipped {completed}/{total} already-filled runs.", flush=True)
    print(f"Running {len(jobs)} runs; total grid size is {total}.", flush=True)

    async def run_one(row: dict[str, object]) -> None:
        nonlocal completed
        async with semaphore:
            row["raw_response"] = ""
            row["KB"] = ""
            row["error"] = ""
            try:
                prompt_text = fill_prompt(
                    str(row["prompt"]),
                    [str(row["premise"])],
                    str(row["hypothesis"]),
                )
                raw = await call_with_retries(
                    client,
                    prompt_text=prompt_text,
                    model=str(row["model"]),
                    temperature=temperature,
                    max_retries=max_retries,
                    request_timeout=request_timeout,
                )
                row["raw_response"] = raw
                parsed = _extract_validated_kb_from_output(raw)
                normalized = normalize_relations(
                    parsed,
                    row_to_problem({key: str(value) for key, value in row.items()}),
                    filter_kb=filter_kb,
                    filtering=filtering,
                )
                row["KB"] = format_kb(normalized)
            except Exception as exc:
                row["error"] = str(exc)

        async with write_lock:
            completed += 1
            if completed % write_every_jobs == 0 or completed == total:
                write_rows(output_csv, OUTPUT_FIELDNAMES, rows)
                print(f"Wrote {output_csv} after {completed}/{total} runs.", flush=True)

    if jobs:
        await asyncio.gather(*(run_one(row) for row in jobs))
    write_rows(output_csv, OUTPUT_FIELDNAMES, rows)


def pairwise_f1(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    counts = relation_counts(left, right)
    if counts.tp == 0 and counts.fp == 0 and counts.fn == 0:
        return 1.0
    return counts.f1


def metric_row(prompt: str, model: str, rows: list[dict[str, str]], repeats: int) -> dict[str, object]:
    by_item: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_item.setdefault(row["ID"], []).append(row)

    valid_items = 0
    all_identical_items = 0
    no_relation_flip_items = 0
    unique_counts = []
    relation_counts_per_run = []
    pair_exact_values = []
    pair_f1_values = []
    valid_runs = 0
    error_runs = 0

    for item_rows in by_item.values():
        item_rows = sorted(item_rows, key=lambda row: int(row["repeat"]))
        error_runs += sum(1 for row in item_rows if row.get("error", "").strip())
        valid = [row for row in item_rows if row.get("KB", "").strip() and not row.get("error", "").strip()]
        valid_runs += len(valid)
        if len(valid) != repeats:
            continue

        parsed = [parse_kb_cell(row["KB"]) for row in valid]
        valid_items += 1
        unique = {kb for kb in parsed}
        unique_counts.append(len(unique))
        relation_counts_per_run.extend(len(kb) for kb in parsed)
        if len(unique) == 1:
            all_identical_items += 1

        has_empty = any(not kb for kb in parsed)
        has_nonempty = any(bool(kb) for kb in parsed)
        if has_empty and has_nonempty:
            no_relation_flip_items += 1

        for left, right in itertools.combinations(parsed, 2):
            pair_exact_values.append(1.0 if left == right else 0.0)
            pair_f1_values.append(pairwise_f1(left, right))

    total_runs = len(rows)
    return {
        "prompt": prompt,
        "model": model,
        "model_slug": model_slug(model),
        "items": len(by_item),
        "repeats": repeats,
        "total_runs": total_runs,
        "valid_runs": valid_runs,
        "error_runs": error_runs,
        "error_rate": error_runs / total_runs if total_runs else math.nan,
        "complete_items": valid_items,
        "all_runs_identical_items": all_identical_items,
        "all_runs_identical_rate": all_identical_items / valid_items if valid_items else math.nan,
        "mean_pairwise_exact": sum(pair_exact_values) / len(pair_exact_values) if pair_exact_values else math.nan,
        "mean_pairwise_kb_f1": sum(pair_f1_values) / len(pair_f1_values) if pair_f1_values else math.nan,
        "no_relation_flip_items": no_relation_flip_items,
        "no_relation_flip_rate": no_relation_flip_items / valid_items if valid_items else math.nan,
        "mean_unique_kb_sets_per_item": sum(unique_counts) / len(unique_counts) if unique_counts else math.nan,
        "avg_relations_per_valid_run": (
            sum(relation_counts_per_run) / len(relation_counts_per_run)
            if relation_counts_per_run
            else math.nan
        ),
    }


def write_metrics(output_rows: list[dict[str, object]], metrics_csv: Path, repeats: int) -> None:
    for row in output_rows:
        if not str(row.get("KB", "")).strip() and not str(row.get("error", "")).strip():
            row["error"] = "missing KB after completed consistency run"

    rows = [{key: str(value) for key, value in row.items()} for row in output_rows]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["prompt"], row["model"]), []).append(row)

    metrics = [
        metric_row(prompt, model, group_rows, repeats)
        for (prompt, model), group_rows in sorted(grouped.items())
    ]
    metrics.sort(
        key=lambda row: (
            -float(row["mean_pairwise_kb_f1"])
            if not math.isnan(float(row["mean_pairwise_kb_f1"]))
            else 1.0,
            row["prompt"],
            row["model"],
        )
    )
    fieldnames = [
        "prompt",
        "model",
        "model_slug",
        "items",
        "repeats",
        "total_runs",
        "valid_runs",
        "error_runs",
        "error_rate",
        "complete_items",
        "all_runs_identical_items",
        "all_runs_identical_rate",
        "mean_pairwise_exact",
        "mean_pairwise_kb_f1",
        "no_relation_flip_items",
        "no_relation_flip_rate",
        "mean_unique_kb_sets_per_item",
        "avg_relations_per_valid_run",
    ]
    write_rows(metrics_csv, fieldnames, metrics)


def write_f1_metrics(
    output_rows: list[dict[str, object]],
    source_rows: list[dict[str, str]],
    reference_columns: list[str],
    f1_metrics_csv: Path,
    f1_summary_csv: Path,
) -> None:
    """Score each prompt/model/repeat group against the human references."""
    source_by_id = {row.get("ID", ""): row for row in source_rows}
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for raw_row in output_rows:
        row = {key: str(value) for key, value in raw_row.items()}
        try:
            repeat = int(row.get("repeat", ""))
        except ValueError:
            continue
        grouped.setdefault((row["prompt"], row["model"], repeat), []).append(row)

    metric_rows: list[dict[str, object]] = []
    for (prompt, model, repeat), prediction_rows in sorted(grouped.items()):
        evaluation_rows: list[dict[str, str]] = []
        error_runs = 0
        for prediction_row in prediction_rows:
            source = source_by_id.get(prediction_row.get("ID", ""), {})
            joined = dict(prediction_row)
            for column in reference_columns:
                joined[column] = source.get(column, "")
            evaluation_rows.append(joined)
            if prediction_row.get("error", "").strip():
                error_runs += 1

        result = evaluate_prediction_column(
            evaluation_rows,
            "KB",
            reference_columns,
            empty_prediction_is_no_relation=False,
            calculate_position_sensitive=True,
            calculate_argument_order_agnostic=True,
        )
        counts = result.selected_counts
        position_counts = result.position_sensitive_counts
        argument_order_agnostic_counts = result.argument_order_agnostic_counts
        assert position_counts is not None
        assert argument_order_agnostic_counts is not None
        exact_rate = (
            result.exact_best_matches / result.evaluated_items
            if result.evaluated_items
            else math.nan
        )
        argument_order_agnostic_exact_rate = (
            result.argument_order_agnostic_exact_best_matches / result.evaluated_items
            if result.evaluated_items
            else math.nan
        )
        metric_rows.append(
            {
                "prompt": prompt,
                "model": model,
                "model_slug": model_slug(model),
                "reference_columns": ";".join(reference_columns),
                "repeat": repeat,
                "total_items": len(prediction_rows),
                "evaluated_items": result.evaluated_items,
                "error_runs": error_runs,
                "skipped_missing_prediction": result.skipped_missing_prediction,
                "skipped_no_reference": result.skipped_no_reference,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "precision": counts.precision,
                "recall": counts.recall,
                "micro_f1": counts.f1,
                "position_sensitive_tp": position_counts.tp,
                "position_sensitive_fp": position_counts.fp,
                "position_sensitive_fn": position_counts.fn,
                "position_sensitive_precision": position_counts.precision,
                "position_sensitive_recall": position_counts.recall,
                "position_sensitive_micro_f1": position_counts.f1,
                "argument_order_agnostic_tp": argument_order_agnostic_counts.tp,
                "argument_order_agnostic_fp": argument_order_agnostic_counts.fp,
                "argument_order_agnostic_fn": argument_order_agnostic_counts.fn,
                "argument_order_agnostic_precision": argument_order_agnostic_counts.precision,
                "argument_order_agnostic_recall": argument_order_agnostic_counts.recall,
                "argument_order_agnostic_micro_f1": argument_order_agnostic_counts.f1,
                "argument_order_agnostic_exact_best_matches": result.argument_order_agnostic_exact_best_matches,
                "argument_order_agnostic_exact_match_rate": argument_order_agnostic_exact_rate,
                "exact_best_matches": result.exact_best_matches,
                "exact_best_match_rate": exact_rate,
                "no_relation_best_matches": result.no_relation_best_matches,
            }
        )

    metric_fieldnames = [
        "prompt",
        "model",
        "model_slug",
        "reference_columns",
        "repeat",
        "total_items",
        "evaluated_items",
        "error_runs",
        "skipped_missing_prediction",
        "skipped_no_reference",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "micro_f1",
        "position_sensitive_tp",
        "position_sensitive_fp",
        "position_sensitive_fn",
        "position_sensitive_precision",
        "position_sensitive_recall",
        "position_sensitive_micro_f1",
        "argument_order_agnostic_tp",
        "argument_order_agnostic_fp",
        "argument_order_agnostic_fn",
        "argument_order_agnostic_precision",
        "argument_order_agnostic_recall",
        "argument_order_agnostic_micro_f1",
        "argument_order_agnostic_exact_best_matches",
        "argument_order_agnostic_exact_match_rate",
        "exact_best_matches",
        "exact_best_match_rate",
        "no_relation_best_matches",
    ]
    write_rows(f1_metrics_csv, metric_fieldnames, metric_rows)

    by_prompt_model: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in metric_rows:
        by_prompt_model.setdefault((str(row["prompt"]), str(row["model"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (prompt, model), rows in sorted(by_prompt_model.items()):
        f1_values = [
            float(row["micro_f1"])
            for row in rows
            if not math.isnan(float(row["micro_f1"]))
        ]
        position_f1_values = [
            float(row["position_sensitive_micro_f1"])
            for row in rows
            if not math.isnan(float(row["position_sensitive_micro_f1"]))
        ]
        argument_order_agnostic_f1_values = [
            float(row["argument_order_agnostic_micro_f1"])
            for row in rows
            if not math.isnan(float(row["argument_order_agnostic_micro_f1"]))
        ]
        argument_order_agnostic_exact_rates = [
            float(row["argument_order_agnostic_exact_match_rate"])
            for row in rows
            if not math.isnan(float(row["argument_order_agnostic_exact_match_rate"]))
        ]
        summary_rows.append(
            {
                "prompt": prompt,
                "model": model,
                "model_slug": model_slug(model),
                "reference_columns": ";".join(reference_columns),
                "repeats_requested": len(rows),
                "repeats_with_f1": len(f1_values),
                "mean_micro_f1": statistics.mean(f1_values) if f1_values else math.nan,
                "sample_stddev_micro_f1": (
                    statistics.stdev(f1_values) if len(f1_values) > 1 else math.nan
                ),
                "min_micro_f1": min(f1_values) if f1_values else math.nan,
                "max_micro_f1": max(f1_values) if f1_values else math.nan,
                "repeats_with_position_sensitive_f1": len(position_f1_values),
                "mean_position_sensitive_micro_f1": (
                    statistics.mean(position_f1_values)
                    if position_f1_values
                    else math.nan
                ),
                "sample_stddev_position_sensitive_micro_f1": (
                    statistics.stdev(position_f1_values)
                    if len(position_f1_values) > 1
                    else math.nan
                ),
                "min_position_sensitive_micro_f1": (
                    min(position_f1_values) if position_f1_values else math.nan
                ),
                "max_position_sensitive_micro_f1": (
                    max(position_f1_values) if position_f1_values else math.nan
                ),
                "repeats_with_argument_order_agnostic_f1": len(
                    argument_order_agnostic_f1_values
                ),
                "mean_argument_order_agnostic_micro_f1": (
                    statistics.mean(argument_order_agnostic_f1_values)
                    if argument_order_agnostic_f1_values
                    else math.nan
                ),
                "sample_stddev_argument_order_agnostic_micro_f1": (
                    statistics.stdev(argument_order_agnostic_f1_values)
                    if len(argument_order_agnostic_f1_values) > 1
                    else math.nan
                ),
                "min_argument_order_agnostic_micro_f1": (
                    min(argument_order_agnostic_f1_values)
                    if argument_order_agnostic_f1_values
                    else math.nan
                ),
                "max_argument_order_agnostic_micro_f1": (
                    max(argument_order_agnostic_f1_values)
                    if argument_order_agnostic_f1_values
                    else math.nan
                ),
                "mean_argument_order_agnostic_exact_match_rate": (
                    statistics.mean(argument_order_agnostic_exact_rates)
                    if argument_order_agnostic_exact_rates
                    else math.nan
                ),
                "sample_stddev_argument_order_agnostic_exact_match_rate": (
                    statistics.stdev(argument_order_agnostic_exact_rates)
                    if len(argument_order_agnostic_exact_rates) > 1
                    else math.nan
                ),
                "min_argument_order_agnostic_exact_match_rate": (
                    min(argument_order_agnostic_exact_rates)
                    if argument_order_agnostic_exact_rates
                    else math.nan
                ),
                "max_argument_order_agnostic_exact_match_rate": (
                    max(argument_order_agnostic_exact_rates)
                    if argument_order_agnostic_exact_rates
                    else math.nan
                ),
                "total_evaluated_items": sum(int(row["evaluated_items"]) for row in rows),
                "total_error_runs": sum(int(row["error_runs"]) for row in rows),
                "total_missing_predictions": sum(
                    int(row["skipped_missing_prediction"]) for row in rows
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -float(row["mean_micro_f1"])
            if not math.isnan(float(row["mean_micro_f1"]))
            else math.inf,
            str(row["prompt"]),
            str(row["model"]),
        )
    )
    summary_fieldnames = [
        "prompt",
        "model",
        "model_slug",
        "reference_columns",
        "repeats_requested",
        "repeats_with_f1",
        "mean_micro_f1",
        "sample_stddev_micro_f1",
        "min_micro_f1",
        "max_micro_f1",
        "repeats_with_position_sensitive_f1",
        "mean_position_sensitive_micro_f1",
        "sample_stddev_position_sensitive_micro_f1",
        "min_position_sensitive_micro_f1",
        "max_position_sensitive_micro_f1",
        "repeats_with_argument_order_agnostic_f1",
        "mean_argument_order_agnostic_micro_f1",
        "sample_stddev_argument_order_agnostic_micro_f1",
        "min_argument_order_agnostic_micro_f1",
        "max_argument_order_agnostic_micro_f1",
        "mean_argument_order_agnostic_exact_match_rate",
        "sample_stddev_argument_order_agnostic_exact_match_rate",
        "min_argument_order_agnostic_exact_match_rate",
        "max_argument_order_agnostic_exact_match_rate",
        "total_evaluated_items",
        "total_error_runs",
        "total_missing_predictions",
    ]
    write_rows(f1_summary_csv, summary_fieldnames, summary_rows)


def write_sample(path: Path, sample_rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    write_rows(path, fieldnames, sample_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument(
        "--sample-csv",
        default=str(DEFAULT_OUTPUT_DIR / "consistency_sample.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_DIR / "consistency_outputs.csv"),
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_OUTPUT_DIR / "consistency_metrics.csv"),
    )
    parser.add_argument(
        "--f1-metrics-csv",
        default=str(DEFAULT_OUTPUT_DIR / "consistency_f1_by_run.csv"),
    )
    parser.add_argument(
        "--f1-summary-csv",
        default=str(DEFAULT_OUTPUT_DIR / "consistency_f1_summary.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reference-columns", nargs="+", default=DEFAULT_REFERENCE_COLUMNS)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--write-every-jobs", type=int, default=40)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-filter-kb", action="store_true")
    parser.add_argument(
        "--filtering-config", type=Path,
        help="JSON object of FilteringConfig settings; defaults to the operational profile.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    sample_path = Path(args.sample_csv)
    output_path = Path(args.output_csv)
    metrics_path = Path(args.metrics_csv)
    f1_metrics_path = Path(args.f1_metrics_csv)
    f1_summary_path = Path(args.f1_summary_csv)
    filtering_path = getattr(args, "filtering_config", None)
    if filtering_path and args.no_filter_kb:
        raise ValueError("--filtering-config cannot be combined with --no-filter-kb")
    filtering = (
        FilteringConfig.model_validate_json(Path(filtering_path).read_text(encoding="utf-8"))
        if filtering_path else FilteringConfig.operational()
    )
    config_json = json.dumps(
        filtering.model_dump(mode="json") if not args.no_filter_kb else None,
        sort_keys=True, separators=(",", ":"),
    )
    existing = load_existing_outputs(output_path) if args.resume else {}
    for row in existing.values():
        if is_done(row) and (
            row.get("filtering_config") != config_json
            or row.get("filtering_version") != FILTERING_PIPELINE_VERSION
        ):
            raise ValueError(
                "Existing outputs have different or missing filtering metadata. "
                "Use a new --output-csv path to preserve historical results."
            )

    fieldnames, rows = read_rows(input_path)
    validate_input_rows(input_path, fieldnames, rows, args.reference_columns)
    sample_rows = choose_balanced_sample(
        rows,
        sample_size=args.sample_size,
        reference_columns=args.reference_columns,
    )
    write_sample(sample_path, sample_rows, fieldnames)

    output_rows = build_output_rows(
        sample_rows,
        args.prompts,
        args.models,
        args.repeats,
        args.reference_columns,
        existing,
    )
    for row in output_rows:
        row["filtering_config"] = config_json
        row["filtering_version"] = FILTERING_PIPELINE_VERSION
    write_rows(output_path, OUTPUT_FIELDNAMES, output_rows)

    if args.prepare_only:
        write_metrics(output_rows, metrics_path, args.repeats)
        write_f1_metrics(
            output_rows,
            sample_rows,
            args.reference_columns,
            f1_metrics_path,
            f1_summary_path,
        )
        print(f"Wrote sample: {sample_path} ({len(sample_rows)} rows)")
        print(f"Wrote output skeleton: {output_path} ({len(output_rows)} runs)")
        print(f"Wrote per-run F1: {f1_metrics_path}")
        print(f"Wrote F1 summary: {f1_summary_path}")
        print("Prepare-only mode: no model calls made.")
        return

    await run_generation(
        output_rows,
        output_csv=output_path,
        provider=args.provider,
        concurrency=args.concurrency,
        write_every_jobs=args.write_every_jobs,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        temperature=args.temperature,
        filter_kb=not args.no_filter_kb,
        filtering=filtering,
    )
    write_metrics(output_rows, metrics_path, args.repeats)
    write_f1_metrics(
        output_rows,
        sample_rows,
        args.reference_columns,
        f1_metrics_path,
        f1_summary_path,
    )
    print(f"Wrote sample: {sample_path} ({len(sample_rows)} rows)")
    print(f"Wrote outputs: {output_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote per-run F1: {f1_metrics_path}")
    print(f"Wrote F1 summary: {f1_summary_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1.")
    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1.")
    if args.write_every_jobs < 1:
        raise SystemExit("--write-every-jobs must be >= 1.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
