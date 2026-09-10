#!/usr/bin/env python3
"""Calculate exact KB inter-annotator agreement for assignment JSON files."""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import re
from pathlib import Path


NO_RELATION_VALUES = {
    "",
    "()",
    "(no, relation)",
    "(no relation)",
    "no relation",
    "(none)",
    "none",
}


def normalize_kb(value: object) -> frozenset[tuple[str, ...]]:
    """Normalize a KB cell into an unordered set of relation tuples."""
    text = str(value or "").strip().lower().replace("?", "")
    if text in NO_RELATION_VALUES:
        return frozenset()

    relations: list[tuple[str, ...]] = []
    for match in re.finditer(r"\(([^()]*)\)", text):
        inner = match.group(1).strip()
        if not inner or inner in {"no, relation", "no relation", "none"}:
            continue
        relations.append(tuple(part.strip() for part in inner.split(",")))

    if relations:
        return frozenset(relations)
    return frozenset({(text,)})


def is_unfilled_annotation(annotator: str, value: object) -> bool:
    """Return whether a row should be treated as absent for agreement."""
    return annotator == "Stefan" and str(value or "").strip() == ""


def label(kb_set: frozenset[tuple[str, ...]]) -> str:
    """Convert a normalized KB set to a stable nominal label."""
    if not kb_set:
        return "NO_RELATION"
    return "; ".join("(" + ", ".join(pair) + ")" for pair in sorted(kb_set))


def normalize_problem_id(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def normalize_gold_label(value: object) -> str:
    return str(value or "").strip().lower()


def build_problem_index(project_root: Path) -> dict[str, dict[str, str]]:
    """Build an ID lookup from the local SICK/SNLI source files."""
    problem_index: dict[str, dict[str, str]] = {}

    sick_files = {
        "train": project_root / "data" / "sick" / "SICK_train.txt",
        "dev": project_root / "data" / "sick" / "SICK_trial.txt",
        "test": project_root / "data" / "sick" / "SICK_test_annotated.txt",
    }
    for split, path in sick_files.items():
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                item_id = str(row.get("pair_ID") or row.get("id") or "").strip()
                if not item_id:
                    continue
                problem_index[item_id] = {
                    "dataset": "sick",
                    "split": split,
                    "premise": row.get("sentence_A", ""),
                    "hypothesis": row.get("sentence_B", ""),
                    "gold_label": normalize_gold_label(
                        row.get("entailment_judgment") or row.get("entailment_label")
                    ),
                }

    snli_files = {
        "train": project_root / "data" / "snli" / "snli_1.0" / "snli_1.0_train.jsonl",
        "dev": project_root / "data" / "snli" / "snli_1.0" / "snli_1.0_dev.jsonl",
        "test": project_root / "data" / "snli" / "snli_1.0" / "snli_1.0_test.jsonl",
    }
    for split, path in snli_files.items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("gold_label") == "-":
                    continue
                item_id = normalize_problem_id(row.get("pairID", ""))
                if not item_id:
                    continue
                problem_index[item_id] = {
                    "dataset": "snli",
                    "split": split,
                    "premise": row.get("sentence1", ""),
                    "hypothesis": row.get("sentence2", ""),
                    "gold_label": normalize_gold_label(row.get("gold_label")),
                }

    return problem_index


def krippendorff_alpha_nominal(units: list[list[str]]) -> float:
    """Krippendorff's alpha for nominal labels."""
    observed_num = 0
    observed_den = 0
    all_ratings: list[str] = []

    for ratings in units:
        ratings = [rating for rating in ratings if rating is not None]
        all_ratings.extend(ratings)
        count = len(ratings)
        if count < 2:
            continue
        counts = collections.Counter(ratings)
        observed_num += count * count - sum(value * value for value in counts.values())
        observed_den += count * (count - 1)

    if observed_den == 0:
        return float("nan")

    observed = observed_num / observed_den
    total = len(all_ratings)
    counts = collections.Counter(all_ratings)
    expected = (total * total - sum(value * value for value in counts.values())) / (
        total * (total - 1)
    )
    if expected == 0:
        return float("nan")
    return 1 - observed / expected


def linear_weighted_cohen_kappa(left: list[int], right: list[int]) -> float:
    """Cohen's kappa with linear weights for ordered relation counts."""
    if len(left) != len(right):
        raise ValueError("Rating lists must have equal lengths.")
    if not left:
        return float("nan")

    highest_label = max(left + right)
    if highest_label == 0:
        return float("nan")

    total = len(left)
    left_counts = collections.Counter(left)
    right_counts = collections.Counter(right)
    observed_weighted = 0.0
    expected_weighted = 0.0
    observed_pairs = collections.Counter(zip(left, right))

    for left_label in range(highest_label + 1):
        for right_label in range(highest_label + 1):
            agreement_weight = 1 - abs(left_label - right_label) / highest_label
            observed_weighted += (
                agreement_weight
                * observed_pairs[left_label, right_label]
                / total
            )
            expected_weighted += agreement_weight * (
                left_counts[left_label] / total
            ) * (right_counts[right_label] / total)

    if expected_weighted == 1:
        return float("nan")
    return (observed_weighted - expected_weighted) / (1 - expected_weighted)


def relation_micro_f1(
    left: list[frozenset[tuple[str, ...]]],
    right: list[frozenset[tuple[str, ...]]],
) -> tuple[float, int, int, int]:
    """Micro F1 over exact normalized KB relations."""
    if len(left) != len(right):
        raise ValueError("Annotation lists must have equal lengths.")

    true_positive = sum(len(a & b) for a, b in zip(left, right))
    false_positive = sum(len(a - b) for a, b in zip(left, right))
    false_negative = sum(len(b - a) for a, b in zip(left, right))
    denominator = 2 * true_positive + false_positive + false_negative
    score = 2 * true_positive / denominator if denominator else float("nan")
    return score, true_positive, false_positive, false_negative


def load_assignments(
    folder: Path,
) -> tuple[
    dict[str, dict[str, frozenset[tuple[str, ...]]]],
    dict[str, dict[str, dict[str, str]]],
    dict[str, int],
    dict[str, list[str]],
]:
    assignments: dict[str, dict[str, frozenset[tuple[str, ...]]]] = {}
    metadata: dict[str, dict[str, dict[str, str]]] = {}
    row_counts: dict[str, int] = {}
    duplicate_ids: dict[str, list[str]] = {}
    for path in sorted(folder.glob("*_assignments.json")):
        annotator = path.stem.replace("_assignments", "")
        rows = json.loads(path.read_text(encoding="utf-8"))
        row_counts[annotator] = len(rows)
        ids = [normalize_problem_id(row["ID"]) for row in rows]
        duplicate_ids[annotator] = [
            item_id for item_id, count in collections.Counter(ids).items() if count > 1
        ]
        assignments[annotator] = {}
        for row in rows:
            if is_unfilled_annotation(annotator, row.get("KB", "")):
                continue
            assignments[annotator][normalize_problem_id(row["ID"])] = normalize_kb(
                row.get("KB", "")
            )
        metadata[annotator] = {
            normalize_problem_id(row["ID"]): {
                "premise": str(row.get("premise", "")),
                "hypothesis": str(row.get("hypothesis", "")),
            }
            for row in rows
        }
    return assignments, metadata, row_counts, duplicate_ids


def annotation_fallback_metadata(
    metadata: dict[str, dict[str, dict[str, str]]],
    annotators: list[str],
    item_id: str,
) -> dict[str, str]:
    for annotator in annotators:
        if item_id in metadata[annotator]:
            return {
                "dataset": "",
                "split": "",
                "premise": metadata[annotator][item_id]["premise"],
                "hypothesis": metadata[annotator][item_id]["hypothesis"],
                "gold_label": "",
            }
    return {
        "dataset": "",
        "split": "",
        "premise": "",
        "hypothesis": "",
        "gold_label": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate exact KB inter-annotator agreement."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="data/annotator_assignments",
        help="Folder containing the tracked *_assignments.json files.",
    )
    parser.add_argument(
        "--csv",
        default="inter_annotator_agreement_overview.csv",
        help="Path for a per-item CSV overview.",
    )
    parser.add_argument(
        "--tsv",
        default="inter_annotator_agreement_overview.tsv",
        help="Path for a per-item TSV overview.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    assignments, metadata, row_counts, duplicate_ids = load_assignments(Path(args.folder))
    problem_index = build_problem_index(project_root)
    if len(assignments) < 2:
        raise SystemExit("Need at least two *_assignments.json files.")

    annotators = list(assignments)
    id_sets = {name: set(items) for name, items in assignments.items()}
    common_ids = sorted(set.intersection(*id_sets.values()))

    print("Annotators:")
    for name in annotators:
        print(
            f"  {name}: {row_counts[name]} rows, "
            f"{len(assignments[name])} unique IDs"
        )

    duplicates = {
        name: ids for name, ids in duplicate_ids.items() if ids
    }
    if duplicates:
        print("\nDuplicate IDs collapsed during ID alignment:")
        for name, ids in duplicates.items():
            print(f"  {name}: {', '.join(ids)}")

    print("\nPairwise exact KB-set agreement:")
    for left, right in itertools.combinations(annotators, 2):
        overlap = sorted(id_sets[left] & id_sets[right])
        matches = sum(assignments[left][item_id] == assignments[right][item_id] for item_id in overlap)
        agreement = matches / len(overlap) if overlap else float("nan")
        print(
            f"  {left}-{right}: {agreement:.3f} "
            f"({matches}/{len(overlap)} shared items)"
        )

    print("\nPairwise linear-weighted Cohen kappa (number of KB relations):")
    for left, right in itertools.combinations(annotators, 2):
        overlap = sorted(id_sets[left] & id_sets[right])
        left_relation_counts = [
            len(assignments[left][item_id]) for item_id in overlap
        ]
        right_relation_counts = [
            len(assignments[right][item_id]) for item_id in overlap
        ]
        kappa = linear_weighted_cohen_kappa(
            left_relation_counts, right_relation_counts
        )
        print(f"  {left}-{right}: {kappa:.3f} ({len(overlap)} shared items)")

    print("\nPairwise micro F1 (exact normalized KB relations):")
    for left, right in itertools.combinations(annotators, 2):
        overlap = sorted(id_sets[left] & id_sets[right])
        score, true_positive, false_positive, false_negative = relation_micro_f1(
            [assignments[left][item_id] for item_id in overlap],
            [assignments[right][item_id] for item_id in overlap],
        )
        print(
            f"  {left}-{right}: {score:.3f} ({len(overlap)} shared items; "
            f"TP={true_positive}, FP={false_positive}, FN={false_negative})"
        )

    all_match_count = sum(
        len({assignments[name][item_id] for name in annotators}) == 1
        for item_id in common_ids
    )
    all_match_agreement = all_match_count / len(common_ids) if common_ids else float("nan")
    alpha_units = [
        [label(assignments[name][item_id]) for name in annotators] for item_id in common_ids
    ]

    print("\nAll-annotator exact KB-set agreement:")
    print(
        f"  Shared by all annotators: {len(common_ids)} items"
    )
    print(
        f"  Exact all-annotator match: {all_match_agreement:.3f} "
        f"({all_match_count}/{len(common_ids)})"
    )
    print(
        "  Krippendorff alpha, nominal exact KB labels: "
        f"{krippendorff_alpha_nominal(alpha_units):.3f}"
    )

    all_ids = sorted(set.union(*id_sets.values()))
    annotator_columns = [f"{name}_KB" for name in annotators]
    pair_columns = [
        f"{left}_{right}_match" for left, right in itertools.combinations(annotators, 2)
    ]
    fieldnames = [
        "ID",
        "premise",
        "hypothesis",
        "label",
        "dataset",
        "split",
        "annotator_count",
        "all_annotators_present",
        "all_present_exact_match",
        *annotator_columns,
        *pair_columns,
    ]
    output_rows = []
    for item_id in all_ids:
        present = [name for name in annotators if item_id in assignments[name]]
        problem = problem_index.get(item_id) or annotation_fallback_metadata(
            metadata, annotators, item_id
        )
        row = {
            "ID": item_id,
            "premise": problem["premise"],
            "hypothesis": problem["hypothesis"],
            "label": problem["gold_label"],
            "dataset": problem["dataset"],
            "split": problem["split"],
            "annotator_count": len(present),
            "all_annotators_present": len(present) == len(annotators),
            "all_present_exact_match": (
                len(present) > 0
                and len({assignments[name][item_id] for name in present}) == 1
            ),
        }
        for name in annotators:
            row[f"{name}_KB"] = (
                label(assignments[name][item_id])
                if item_id in assignments[name]
                else ""
            )
        for left, right in itertools.combinations(annotators, 2):
            row[f"{left}_{right}_match"] = (
                assignments[left][item_id] == assignments[right][item_id]
                if item_id in assignments[left] and item_id in assignments[right]
                else ""
            )
        output_rows.append(row)

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    tsv_path = Path(args.tsv)
    with tsv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"\nWrote CSV overview: {csv_path}")
    print(f"Wrote TSV overview: {tsv_path}")


if __name__ == "__main__":
    main()
