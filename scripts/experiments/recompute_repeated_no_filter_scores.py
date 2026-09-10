#!/usr/bin/env python3
"""Recompute repeated-run KBs and scores from raw responses without filtering.

This script performs no model or network calls. It reparses each saved raw
response, converts ``entails`` to ``isa_wn`` through the existing parser, skips
the generation-time normalization/filtering pipeline, and regenerates quality
and stability metrics in separate output files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from calculate_multi_reference_f1 import parse_kb_cell
from kbprojection.llm import _extract_validated_kb_from_output
from scripts.experiments.run_repeated_multi_reference_experiment import (
    DEFAULT_REFERENCE_COLUMNS,
    OUTPUT_FIELDNAMES,
    format_kb,
    read_rows,
    write_f1_metrics,
    write_metrics,
    write_rows,
)

DEFAULT_RESULTS_DIR = (
    REPOSITORY_ROOT / "experiment_results" / "lasha_all362_5runs"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_outputs.csv"
        ),
    )
    parser.add_argument(
        "--sample-csv",
        default=str(REPOSITORY_ROOT / "data" / "all_usable_items_362.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_no_filter_outputs.csv"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_no_filter_stability.csv"
        ),
    )
    parser.add_argument(
        "--f1-metrics-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_no_filter_f1_by_run.csv"
        ),
    )
    parser.add_argument(
        "--f1-summary-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_no_filter_f1_summary.csv"
        ),
    )
    parser.add_argument(
        "--filtered-f1-summary-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_f1_summary.csv"
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        default=str(
            DEFAULT_RESULTS_DIR
            / "small_medium_lasha_all362_5runs_filtered_vs_no_filter.csv"
        ),
    )
    parser.add_argument(
        "--reference-columns",
        nargs="+",
        default=DEFAULT_REFERENCE_COLUMNS,
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    sample_path = Path(args.sample_csv)
    output_path = Path(args.output_csv)
    metrics_path = Path(args.metrics_csv)
    f1_metrics_path = Path(args.f1_metrics_csv)
    f1_summary_path = Path(args.f1_summary_csv)
    filtered_f1_summary_path = Path(args.filtered_f1_summary_csv)
    comparison_path = Path(args.comparison_csv)

    _, original_rows = read_rows(input_path)
    sample_fieldnames, source_rows = read_rows(sample_path)
    missing_references = [
        column
        for column in args.reference_columns
        if column not in sample_fieldnames
    ]
    if missing_references:
        raise SystemExit(
            "Missing reference column(s) in sample CSV: "
            + ", ".join(missing_references)
        )

    recomputed_rows: list[dict[str, object]] = []
    changed_kbs = 0
    reparsed_rows = 0
    missing_raw_rows = 0
    parse_error_rows = 0

    for original in original_rows:
        row: dict[str, object] = {
            field: original.get(field, "") for field in OUTPUT_FIELDNAMES
        }
        raw_response = str(original.get("raw_response", "")).strip()
        original_kb = str(original.get("KB", "")).strip()

        if not raw_response:
            row["KB"] = ""
            if not str(original.get("error", "")).strip():
                row["error"] = "missing raw response during no-filter recomputation"
            missing_raw_rows += 1
        else:
            try:
                parsed = _extract_validated_kb_from_output(raw_response)
                row["KB"] = format_kb(parsed)
                row["error"] = ""
                reparsed_rows += 1
            except Exception as exc:
                row["KB"] = ""
                row["error"] = f"no-filter parse error: {exc}"
                parse_error_rows += 1

        if str(row["KB"]).strip() != original_kb:
            changed_kbs += 1
        recomputed_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_path, OUTPUT_FIELDNAMES, recomputed_rows)
    write_metrics(recomputed_rows, metrics_path, args.repeats)
    write_f1_metrics(
        recomputed_rows,
        source_rows,
        args.reference_columns,
        f1_metrics_path,
        f1_summary_path,
    )

    _, filtered_summary = read_rows(filtered_f1_summary_path)
    _, no_filter_summary = read_rows(f1_summary_path)
    filtered_by_key = {
        (row["prompt"], row["model"]): row for row in filtered_summary
    }
    comparison_rows: list[dict[str, object]] = []
    for no_filter_row in no_filter_summary:
        key = (no_filter_row["prompt"], no_filter_row["model"])
        filtered_row = filtered_by_key.get(key)
        if filtered_row is None:
            raise SystemExit(
                "No filtered summary row for "
                f"prompt={key[0]!r}, model={key[1]!r}"
            )
        filtered_f1 = float(filtered_row["mean_micro_f1"])
        no_filter_f1 = float(no_filter_row["mean_micro_f1"])
        filtered_position_f1 = float(
            filtered_row["mean_position_sensitive_micro_f1"]
        )
        no_filter_position_f1 = float(
            no_filter_row["mean_position_sensitive_micro_f1"]
        )
        comparison_rows.append(
            {
                "prompt": key[0],
                "model": key[1],
                "filtered_mean_micro_f1": filtered_f1,
                "no_filter_mean_micro_f1": no_filter_f1,
                "no_filter_minus_filtered_micro_f1": (
                    no_filter_f1 - filtered_f1
                ),
                "filtered_mean_position_sensitive_micro_f1": (
                    filtered_position_f1
                ),
                "no_filter_mean_position_sensitive_micro_f1": (
                    no_filter_position_f1
                ),
                "no_filter_minus_filtered_position_sensitive_micro_f1": (
                    no_filter_position_f1 - filtered_position_f1
                ),
                "no_filter_total_evaluated_items": no_filter_row[
                    "total_evaluated_items"
                ],
                "no_filter_total_error_runs": no_filter_row[
                    "total_error_runs"
                ],
                "no_filter_total_missing_predictions": no_filter_row[
                    "total_missing_predictions"
                ],
            }
        )
    comparison_rows.sort(
        key=lambda row: float(row["no_filter_mean_micro_f1"]),
        reverse=True,
    )
    comparison_fieldnames = [
        "prompt",
        "model",
        "filtered_mean_micro_f1",
        "no_filter_mean_micro_f1",
        "no_filter_minus_filtered_micro_f1",
        "filtered_mean_position_sensitive_micro_f1",
        "no_filter_mean_position_sensitive_micro_f1",
        "no_filter_minus_filtered_position_sensitive_micro_f1",
        "no_filter_total_evaluated_items",
        "no_filter_total_error_runs",
        "no_filter_total_missing_predictions",
    ]
    write_rows(comparison_path, comparison_fieldnames, comparison_rows)

    unique_keys = {
        (
            str(row.get("ID", "")),
            str(row.get("prompt", "")),
            str(row.get("model", "")),
            str(row.get("repeat", "")),
        )
        for row in recomputed_rows
    }
    nonblank_kbs = sum(
        bool(str(row.get("KB", "")).strip()) for row in recomputed_rows
    )
    explicit_no_relation = sum(
        not parse_kb_cell(row.get("KB", ""))
        and str(row.get("KB", "")).strip().upper() == "NO_RELATION"
        for row in recomputed_rows
    )

    print(f"Input rows: {len(original_rows)}")
    print(f"Unique item/prompt/model/repeat keys: {len(unique_keys)}")
    print(f"Successfully reparsed raw responses: {reparsed_rows}")
    print(f"Rows without raw responses: {missing_raw_rows}")
    print(f"No-filter parse errors: {parse_error_rows}")
    print(f"KB values changed versus filtered input: {changed_kbs}")
    print(f"Nonblank recomputed KB cells: {nonblank_kbs}")
    print(f"Explicit NO_RELATION cells: {explicit_no_relation}")
    print(f"Wrote no-filter outputs: {output_path}")
    print(f"Wrote no-filter stability: {metrics_path}")
    print(f"Wrote no-filter F1 by run: {f1_metrics_path}")
    print(f"Wrote no-filter F1 summary: {f1_summary_path}")
    print(f"Wrote filtered/no-filter comparison: {comparison_path}")


if __name__ == "__main__":
    main()
