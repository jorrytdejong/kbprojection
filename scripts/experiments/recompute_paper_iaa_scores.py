#!/usr/bin/env python3
"""Recompute the paper's post-adjudication IAA scores without network calls.

The paper scores each original annotator against the best of the other three
original annotators, using the final 362 retained items.  ``Alternative_KB``
is deliberately excluded: it is an adjudication-created fourth valid variant,
not an independent annotator submission.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from calculate_multi_reference_f1 import evaluate_prediction_column, load_rows

ANNOTATORS = ("Ettore", "Jorryt", "Lasha", "Stefan")
PAPER_EXACT_PERCENT = {
    "Ettore": 75.7,
    "Jorryt": 84.5,
    "Lasha": 80.8,
    "Stefan": 82.8,
}
PAPER_MICRO_F1_PERCENT = {
    "Ettore": 81.5,
    "Jorryt": 86.0,
    "Lasha": 85.9,
    "Stefan": 86.2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iaa-csv",
        default=str(
            REPOSITORY_ROOT
            / "data"
            / "annotator_agreement"
            / "iaa_overview_edit.csv"
        ),
        help="Edited/adjudicated IAA overview exported from the project sheet.",
    )
    parser.add_argument(
        "--final-items-csv",
        default=str(REPOSITORY_ROOT / "data" / "all_usable_items_362.csv"),
        help="Canonical CSV that defines the 362 final retained IDs.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(
            REPOSITORY_ROOT
            / "experiment_results"
            / "iaa"
            / "paper_table2_iaa_scores.csv"
        ),
        help="Destination for the recomputed per-annotator scores.",
    )
    return parser


def read_final_ids(path: Path) -> set[str]:
    _, rows = load_rows(path)
    ids = {str(row.get("ID", "")).strip() for row in rows}
    ids.discard("")
    if len(ids) != 362:
        raise SystemExit(f"Expected 362 unique IDs in {path}, found {len(ids)}.")
    return ids


def main() -> None:
    args = build_parser().parse_args()
    iaa_path = Path(args.iaa_csv)
    final_items_path = Path(args.final_items_csv)
    summary_path = Path(args.summary_csv)

    _, rows = load_rows(iaa_path)
    final_ids = read_final_ids(final_items_path)
    retained_rows = [
        row for row in rows if str(row.get("ID", "")).strip() in final_ids
    ]
    retained_ids = [str(row["ID"]).strip() for row in retained_rows]
    if len(retained_rows) != 362 or len(set(retained_ids)) != 362:
        raise SystemExit(
            "Expected exactly one edited IAA row for each of the 362 final IDs; "
            f"found {len(retained_rows)} rows and {len(set(retained_ids))} IDs."
        )

    summary_rows: list[dict[str, object]] = []
    for annotator in ANNOTATORS:
        prediction_column = f"{annotator}_KB"
        reference_columns = [
            f"{other}_KB" for other in ANNOTATORS if other != annotator
        ]
        result = evaluate_prediction_column(
            retained_rows,
            prediction_column,
            reference_columns,
            empty_prediction_is_no_relation=False,
        )
        counts = result.selected_counts
        exact_rate = result.exact_best_matches / result.evaluated_items
        calculated_exact_percent = 100 * exact_rate
        calculated_micro_f1_percent = 100 * counts.f1
        summary_rows.append(
            {
                "annotator": annotator,
                "evaluated_items": result.evaluated_items,
                "exact_best_matches": result.exact_best_matches,
                "calculated_exact_match_rate": exact_rate,
                "calculated_exact_match_percent": calculated_exact_percent,
                "calculated_micro_f1_percent": calculated_micro_f1_percent,
                "micro_f1_tp": counts.tp,
                "micro_f1_fp": counts.fp,
                "micro_f1_fn": counts.fn,
                "paper_exact_match_percent": PAPER_EXACT_PERCENT[annotator],
                "paper_micro_f1_percent": PAPER_MICRO_F1_PERCENT[annotator],
            }
        )
        print(
            f"{annotator}: n={result.evaluated_items}, "
            f"exact={calculated_exact_percent:.1f}% "
            f"(paper {PAPER_EXACT_PERCENT[annotator]:.1f}%), "
            f"micro-F1={calculated_micro_f1_percent:.1f}% "
            f"(paper {PAPER_MICRO_F1_PERCENT[annotator]:.1f}%)"
        )

    mean_exact = sum(row["calculated_exact_match_percent"] for row in summary_rows) / 4
    mean_micro_f1 = sum(row["calculated_micro_f1_percent"] for row in summary_rows) / 4
    print(f"Mean: exact={mean_exact:.1f}%, micro-F1={mean_micro_f1:.1f}%")
    print(
        "Note: Ettore's exact rate is 205/271 = 75.645...%, which displays "
        "as 75.6% at one decimal place; Table 2 prints 75.7%."
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
