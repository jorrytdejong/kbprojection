#!/usr/bin/env python3
"""Multi-reference KB evaluation for annotator agreement overview CSV files.

The main metric mirrors multi-reference machine-translation evaluation:
compare one prediction KB against every available reference KB for the same
item, keep the best matching reference for that item, then micro-average the
selected TP/FP/FN counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from calculate_inter_annotator_agreement import normalize_kb


@dataclass(frozen=True)
class ScoringConfig:
    """Evaluation-only normalization settings.

    ``argument_order_agnostic`` is for callers which select that representation
    as their primary metric (the four-condition replay).  The overview CLI keeps
    its historical directed primary score and exposes order agnosticism as the
    separate diagnostic selected by ``calculate_argument_order_agnostic``.
    """

    lemmatize: bool = False
    argument_order_agnostic: bool = False


def prepare_scoring_resources(configs):
    """Verify NLP data only when contextual lemmatization is requested."""
    if not any(config.lemmatize for config in configs):
        return {}
    from kbprojection.filtering_context import prepare_filtering_resources
    from kbprojection.models import FilteringConfig
    return prepare_filtering_resources([
        FilteringConfig.evaluation_baseline(
            leading_preposition_mode="off", lemmatization_mode="replacement")
    ])


class ScoringContext:
    """Symmetric, lazy contextual POS representation for one P/H pair."""

    def __init__(self, premise="", hypothesis=""):
        self.premise, self.hypothesis = premise, hypothesis
        self._nlp = None
        self.argument = lru_cache(maxsize=None)(self._argument)

    def _argument(self, argument):
        # Keep the standard directed scorer importable with only the stdlib.
        from kbprojection.filtering_context import FilteringContext

        if self._nlp is None:
            self._nlp = FilteringContext(self.premise, self.hypothesis, offline=True)
        context = self._nlp
        tokens = tuple(token.lower() for token in context.tokens(argument))
        if not tokens:
            return argument, "empty"
        mapping = {"J": "a", "V": "v", "N": "n", "R": "r"}
        matches = set()
        for sentence in (self.premise, self.hypothesis):
            tagged = context.tags(sentence)
            words = tuple(word.lower() for word, _ in tagged)
            for index in range(len(words) - len(tokens) + 1):
                if words[index:index + len(tokens)] == tokens:
                    matches.add(tuple(mapping.get(tag[:1]) for _, tag in tagged[index:index + len(tokens)]))
        if not matches:
            return argument, "absent"
        if len(matches) != 1:
            return argument, "ambiguous"
        lemmas = tuple(context.lemma(token, pos) if pos else token
                       for token, pos in zip(tokens, next(iter(matches))))
        value = " ".join(lemmas) if lemmas != tokens else argument
        return value, "changed" if value != argument else "unchanged"

    def relation(self, relation, config):
        if config.lemmatize:
            relation = tuple(self.argument(argument)[0] for argument in relation)
        if config.argument_order_agnostic:
            relation = canonicalize_relation_arguments(relation)
        return relation

    def kb(self, relations, config):
        return frozenset(self.relation(relation, config) for relation in relations)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else float("nan")

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else float("nan")

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else float("nan")


@dataclass
class EvaluationResult:
    prediction_column: str
    reference_columns: list[str]
    selected_counts: Counts
    evaluated_items: int = 0
    skipped_missing_prediction: int = 0
    skipped_no_reference: int = 0
    exact_best_matches: int = 0
    no_relation_best_matches: int = 0
    position_sensitive_counts: Counts | None = None
    argument_order_agnostic_counts: Counts | None = None
    argument_order_agnostic_exact_best_matches: int = 0
    scoring: ScoringConfig = ScoringConfig()


def is_blank(value: object) -> bool:
    return str(value or "").strip() == ""


def parse_kb_cell(value: object) -> frozenset[tuple[str, ...]]:
    text = str(value or "").strip()
    if text.upper() == "NO_RELATION":
        return frozenset()
    return normalize_kb(text)


def parse_kb_sequence(value: object) -> tuple[tuple[str, ...], ...]:
    """Normalize a KB cell while preserving the written relation order."""
    text = str(value or "").strip().lower().replace("?", "")
    if text.upper() == "NO_RELATION" or not normalize_kb(text):
        return ()

    relations: list[tuple[str, ...]] = []
    for match in re.finditer(r"\(([^()]*)\)", text):
        inner = match.group(1).strip()
        if not inner or inner in {"no, relation", "no relation", "none"}:
            continue
        relations.append(tuple(part.strip() for part in inner.split(",")))

    if relations:
        return tuple(relations)
    return ((text,),)


def relation_counts(
    prediction: frozenset[tuple[str, ...]],
    reference: frozenset[tuple[str, ...]],
) -> Counts:
    return Counts(
        tp=len(prediction & reference),
        fp=len(prediction - reference),
        fn=len(reference - prediction),
    )


def canonicalize_relation_arguments(
    relation: tuple[str, ...],
) -> tuple[str, ...]:
    """Make a two-argument relation insensitive to its argument direction.

    This is a diagnostic comparison only.  The primary metric keeps relation
    direction intact because it is semantically meaningful for lexical
    entailment relations such as ``isa_wn(dog, animal)``.
    """
    if len(relation) != 2:
        return relation
    return tuple(sorted(relation))


def argument_order_agnostic_relation_counts(
    prediction: frozenset[tuple[str, ...]],
    reference: frozenset[tuple[str, ...]],
) -> Counts:
    """Compare relation sets after canonicalizing the two arguments."""
    canonical_prediction = frozenset(
        canonicalize_relation_arguments(relation) for relation in prediction
    )
    canonical_reference = frozenset(
        canonicalize_relation_arguments(relation) for relation in reference
    )
    return relation_counts(canonical_prediction, canonical_reference)


def position_sensitive_relation_counts(
    prediction: tuple[tuple[str, ...], ...],
    reference: tuple[tuple[str, ...], ...],
) -> Counts:
    """Count exact relation matches at the same sequence position.

    A mismatch at a shared position contributes one FP and one FN. Relations
    extending beyond the shorter sequence contribute an FP (prediction) or FN
    (reference).
    """
    matching_positions = sum(
        predicted_relation == reference_relation
        for predicted_relation, reference_relation in zip(prediction, reference)
    )
    return Counts(
        tp=matching_positions,
        fp=len(prediction) - matching_positions,
        fn=len(reference) - matching_positions,
    )


def item_selection_score(counts: Counts) -> float:
    """F1 used only for choosing the best reference for one item.

    Empty prediction plus empty reference is a perfect item-level match, but it
    contributes no TP/FP/FN to relation-level micro-F1.
    """
    if counts.tp == 0 and counts.fp == 0 and counts.fn == 0:
        return 1.0
    return counts.f1


def format_score(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def sort_score(value: float) -> float:
    return -1.0 if math.isnan(value) else value


def kb_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.endswith("_KB")]


def generated_llm_kb_columns(fieldnames: list[str]) -> list[str]:
    return [
        name
        for name in fieldnames
        if name.startswith("LLM__") and name.endswith("_KB")
    ]


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        return reader.fieldnames, list(reader)


def evaluate_prediction_column(
    rows: list[dict[str, str]],
    prediction_column: str,
    reference_columns: list[str],
    *,
    empty_prediction_is_no_relation: bool,
    calculate_position_sensitive: bool = False,
    calculate_argument_order_agnostic: bool = False,
    details_path: Path | None = None,
    scoring: ScoringConfig = ScoringConfig(),
) -> EvaluationResult:
    result = EvaluationResult(
        prediction_column=prediction_column,
        reference_columns=reference_columns,
        selected_counts=Counts(),
        scoring=scoring,
    )
    detail_rows: list[dict[str, object]] = []
    if calculate_position_sensitive:
        result.position_sensitive_counts = Counts()
    if calculate_argument_order_agnostic:
        result.argument_order_agnostic_counts = Counts()

    for row in rows:
        context = ScoringContext(row.get("premise", ""), row.get("hypothesis", ""))
        raw_prediction = row.get(prediction_column, "")
        if is_blank(raw_prediction) and not empty_prediction_is_no_relation:
            result.skipped_missing_prediction += 1
            continue

        references = [
            (column, context.kb(parse_kb_cell(row.get(column, "")), scoring))
            for column in reference_columns
            if not is_blank(row.get(column, ""))
        ]
        if not references:
            result.skipped_no_reference += 1
            continue

        prediction = context.kb(parse_kb_cell(raw_prediction), scoring)
        scored_references = []
        for column, reference in references:
            counts = relation_counts(prediction, reference)
            scored_references.append((item_selection_score(counts), column, reference, counts))

        best_score, best_column, best_reference, best_counts = max(
            scored_references,
            key=lambda entry: (
                entry[0],
                entry[3].tp,
                -entry[3].fp,
                -entry[3].fn,
                entry[1],
            ),
        )
        result.selected_counts.add(best_counts)

        argument_order_agnostic_best_column = ""
        argument_order_agnostic_best_score = float("nan")
        argument_order_agnostic_best_counts: Counts | None = None
        if calculate_argument_order_agnostic:
            argument_order_agnostic_scored_references = []
            for column, reference in references:
                counts = argument_order_agnostic_relation_counts(
                    prediction, reference
                )
                argument_order_agnostic_scored_references.append(
                    (item_selection_score(counts), column, counts)
                )
            (
                argument_order_agnostic_best_score,
                argument_order_agnostic_best_column,
                argument_order_agnostic_best_counts,
            ) = max(
                argument_order_agnostic_scored_references,
                key=lambda entry: (
                    entry[0],
                    entry[2].tp,
                    -entry[2].fp,
                    -entry[2].fn,
                    entry[1],
                ),
            )
            assert result.argument_order_agnostic_counts is not None
            result.argument_order_agnostic_counts.add(
                argument_order_agnostic_best_counts
            )
            if (
                argument_order_agnostic_best_counts.fp == 0
                and argument_order_agnostic_best_counts.fn == 0
            ):
                result.argument_order_agnostic_exact_best_matches += 1

        position_best_column = ""
        position_best_score = float("nan")
        position_best_counts: Counts | None = None
        if calculate_position_sensitive:
            prediction_sequence = tuple(context.relation(r, scoring) for r in parse_kb_sequence(raw_prediction))
            position_scored_references = []
            for column, _ in references:
                reference_sequence = tuple(context.relation(r, scoring) for r in parse_kb_sequence(row.get(column, "")))
                counts = position_sensitive_relation_counts(
                    prediction_sequence, reference_sequence
                )
                position_scored_references.append(
                    (item_selection_score(counts), column, counts)
                )
            position_best_score, position_best_column, position_best_counts = max(
                position_scored_references,
                key=lambda entry: (
                    entry[0],
                    entry[2].tp,
                    -entry[2].fp,
                    -entry[2].fn,
                    entry[1],
                ),
            )
            assert result.position_sensitive_counts is not None
            result.position_sensitive_counts.add(position_best_counts)
        result.evaluated_items += 1
        if prediction == best_reference:
            result.exact_best_matches += 1
            if not prediction:
                result.no_relation_best_matches += 1

        detail_rows.append(
            {
                "ID": row.get("ID", ""),
                "prediction_column": prediction_column,
                "best_reference_column": best_column,
                "best_item_f1": best_score,
                "tp": best_counts.tp,
                "fp": best_counts.fp,
                "fn": best_counts.fn,
                "prediction_kb": raw_prediction,
                "best_reference_kb": row.get(best_column, ""),
                "position_sensitive_best_reference_column": position_best_column,
                "position_sensitive_best_item_f1": position_best_score,
                "position_sensitive_tp": position_best_counts.tp if position_best_counts else "",
                "position_sensitive_fp": position_best_counts.fp if position_best_counts else "",
                "position_sensitive_fn": position_best_counts.fn if position_best_counts else "",
                "argument_order_agnostic_best_reference_column": argument_order_agnostic_best_column,
                "argument_order_agnostic_best_item_f1": argument_order_agnostic_best_score,
                "argument_order_agnostic_tp": argument_order_agnostic_best_counts.tp if argument_order_agnostic_best_counts else "",
                "argument_order_agnostic_fp": argument_order_agnostic_best_counts.fp if argument_order_agnostic_best_counts else "",
                "argument_order_agnostic_fn": argument_order_agnostic_best_counts.fn if argument_order_agnostic_best_counts else "",
                "argument_order_agnostic_exact_match": (
                    argument_order_agnostic_best_counts is not None
                    and argument_order_agnostic_best_counts.fp == 0
                    and argument_order_agnostic_best_counts.fn == 0
                ),
            }
        )

    if details_path is not None:
        with details_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "ID",
                    "prediction_column",
                    "best_reference_column",
                    "best_item_f1",
                    "tp",
                    "fp",
                    "fn",
                    "prediction_kb",
                    "best_reference_kb",
                    "position_sensitive_best_reference_column",
                    "position_sensitive_best_item_f1",
                    "position_sensitive_tp",
                    "position_sensitive_fp",
                    "position_sensitive_fn",
                    "argument_order_agnostic_best_reference_column",
                    "argument_order_agnostic_best_item_f1",
                    "argument_order_agnostic_tp",
                    "argument_order_agnostic_fp",
                    "argument_order_agnostic_fn",
                    "argument_order_agnostic_exact_match",
                ],
            )
            writer.writeheader()
            writer.writerows(detail_rows)

    return result


def evaluate_global_best_reference(
    rows: list[dict[str, str]],
    prediction_column: str,
    reference_columns: list[str],
    *,
    empty_prediction_is_no_relation: bool,
    scoring: ScoringConfig = ScoringConfig(),
) -> list[tuple[str, Counts, int]]:
    results: list[tuple[str, Counts, int]] = []
    for reference_column in reference_columns:
        counts = Counts()
        evaluated_items = 0
        for row in rows:
            raw_prediction = row.get(prediction_column, "")
            raw_reference = row.get(reference_column, "")
            if is_blank(raw_prediction) and not empty_prediction_is_no_relation:
                continue
            if is_blank(raw_reference):
                continue
            context = ScoringContext(row.get("premise", ""), row.get("hypothesis", ""))
            counts.add(relation_counts(context.kb(parse_kb_cell(raw_prediction), scoring),
                                       context.kb(parse_kb_cell(raw_reference), scoring)))
            evaluated_items += 1
        results.append((reference_column, counts, evaluated_items))
    return sorted(results, key=lambda entry: (-sort_score(entry[1].f1), entry[0]))


def print_result(result: EvaluationResult) -> None:
    counts = result.selected_counts
    print(f"\n{result.prediction_column} vs best of {', '.join(result.reference_columns)}")
    print(f"  scoring: lemmatize_both={result.scoring.lemmatize}, argument_order_agnostic={result.scoring.argument_order_agnostic}")
    print(
        "  multi_reference_micro_f1 "
        f"P={format_score(counts.precision)} "
        f"R={format_score(counts.recall)} "
        f"F1={format_score(counts.f1)} "
        f"(TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
    )
    exact_rate = result.exact_best_matches / result.evaluated_items if result.evaluated_items else float("nan")
    print(
        "  items "
        f"evaluated={result.evaluated_items}, "
        f"exact_best_match={format_score(exact_rate)} "
        f"({result.exact_best_matches}/{result.evaluated_items}), "
        f"no_relation_best_matches={result.no_relation_best_matches}"
    )
    print(
        "  skipped "
        f"missing_prediction={result.skipped_missing_prediction}, "
        f"no_available_reference={result.skipped_no_reference}"
    )
    if result.position_sensitive_counts is not None:
        position_counts = result.position_sensitive_counts
        print(
            "  position_sensitive_relation_sequence_micro_f1 "
            f"P={format_score(position_counts.precision)} "
            f"R={format_score(position_counts.recall)} "
            f"F1={format_score(position_counts.f1)} "
            f"(TP={position_counts.tp}, FP={position_counts.fp}, "
            f"FN={position_counts.fn})"
        )
    if result.argument_order_agnostic_counts is not None:
        argument_order_agnostic_counts = result.argument_order_agnostic_counts
        argument_order_agnostic_exact_rate = (
            result.argument_order_agnostic_exact_best_matches / result.evaluated_items
            if result.evaluated_items
            else float("nan")
        )
        print(
            "  argument_order_agnostic_relation_set_micro_f1 "
            f"P={format_score(argument_order_agnostic_counts.precision)} "
            f"R={format_score(argument_order_agnostic_counts.recall)} "
            f"F1={format_score(argument_order_agnostic_counts.f1)} "
            f"(TP={argument_order_agnostic_counts.tp}, "
            f"FP={argument_order_agnostic_counts.fp}, "
            f"FN={argument_order_agnostic_counts.fn})"
        )
        print(
            "  argument_order_agnostic_exact_match "
            f"{format_score(argument_order_agnostic_exact_rate)} "
            f"({result.argument_order_agnostic_exact_best_matches}/"
            f"{result.evaluated_items})"
        )


def print_global_reference_results(
    prediction_column: str,
    global_results: list[tuple[str, Counts, int]],
) -> None:
    print(f"  best_single_reference_for_{prediction_column}:")
    for reference_column, counts, evaluated_items in global_results:
        print(
            f"    {reference_column}: "
            f"F1={format_score(counts.f1)} "
            f"P={format_score(counts.precision)} "
            f"R={format_score(counts.recall)} "
            f"(items={evaluated_items}, TP={counts.tp}, FP={counts.fp}, FN={counts.fn})"
        )


def write_summary(path: Path, results: list[EvaluationResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prediction_column",
                "reference_columns",
                "lemmatize_both",
                "argument_order_agnostic",
                "evaluated_items",
                "precision",
                "recall",
                "micro_f1",
                "tp",
                "fp",
                "fn",
                "exact_best_matches",
                "exact_best_match_rate",
                "no_relation_best_matches",
                "skipped_missing_prediction",
                "skipped_no_reference",
                "position_sensitive_precision",
                "position_sensitive_recall",
                "position_sensitive_micro_f1",
                "position_sensitive_tp",
                "position_sensitive_fp",
                "position_sensitive_fn",
                "argument_order_agnostic_precision",
                "argument_order_agnostic_recall",
                "argument_order_agnostic_micro_f1",
                "argument_order_agnostic_tp",
                "argument_order_agnostic_fp",
                "argument_order_agnostic_fn",
                "argument_order_agnostic_exact_best_matches",
                "argument_order_agnostic_exact_match_rate",
            ],
        )
        writer.writeheader()
        for result in results:
            counts = result.selected_counts
            position_counts = result.position_sensitive_counts
            argument_order_agnostic_counts = result.argument_order_agnostic_counts
            exact_rate = result.exact_best_matches / result.evaluated_items if result.evaluated_items else float("nan")
            argument_order_agnostic_exact_rate = (
                result.argument_order_agnostic_exact_best_matches
                / result.evaluated_items
                if result.evaluated_items
                else float("nan")
            )
            writer.writerow(
                {
                    "prediction_column": result.prediction_column,
                    "reference_columns": ";".join(result.reference_columns),
                    "lemmatize_both": result.scoring.lemmatize,
                    "argument_order_agnostic": result.scoring.argument_order_agnostic,
                    "evaluated_items": result.evaluated_items,
                    "precision": counts.precision,
                    "recall": counts.recall,
                    "micro_f1": counts.f1,
                    "tp": counts.tp,
                    "fp": counts.fp,
                    "fn": counts.fn,
                    "exact_best_matches": result.exact_best_matches,
                    "exact_best_match_rate": exact_rate,
                    "no_relation_best_matches": result.no_relation_best_matches,
                    "skipped_missing_prediction": result.skipped_missing_prediction,
                    "skipped_no_reference": result.skipped_no_reference,
                    "position_sensitive_precision": position_counts.precision if position_counts else "",
                    "position_sensitive_recall": position_counts.recall if position_counts else "",
                    "position_sensitive_micro_f1": position_counts.f1 if position_counts else "",
                    "position_sensitive_tp": position_counts.tp if position_counts else "",
                    "position_sensitive_fp": position_counts.fp if position_counts else "",
                    "position_sensitive_fn": position_counts.fn if position_counts else "",
                    "argument_order_agnostic_precision": argument_order_agnostic_counts.precision if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_recall": argument_order_agnostic_counts.recall if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_micro_f1": argument_order_agnostic_counts.f1 if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_tp": argument_order_agnostic_counts.tp if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_fp": argument_order_agnostic_counts.fp if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_fn": argument_order_agnostic_counts.fn if argument_order_agnostic_counts else "",
                    "argument_order_agnostic_exact_best_matches": (
                        result.argument_order_agnostic_exact_best_matches
                        if argument_order_agnostic_counts
                        else ""
                    ),
                    "argument_order_agnostic_exact_match_rate": (
                        argument_order_agnostic_exact_rate
                        if argument_order_agnostic_counts
                        else ""
                    ),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute multi-reference KB micro-F1 from an agreement overview CSV."
    )
    parser.add_argument("--lemmatize", action="store_true",
                        help="Compare POS lemmas on BOTH predictions and references. Requires premise/hypothesis columns and installed NLTK data; ambiguous/absent spans remain unchanged.")
    parser.add_argument(
        "--csv",
        default="annotator agreement - inter_annotator_agreement_overview.csv",
        help="Input agreement overview CSV.",
    )
    parser.add_argument(
        "--prediction-column",
        help=(
            "Column to evaluate as the system/LLM KB. If omitted with "
            "--reference-columns, all LLM__*_KB columns are evaluated. "
            "If omitted without --reference-columns, every *_KB column is "
            "evaluated against the others."
        ),
    )
    parser.add_argument(
        "--reference-columns",
        nargs="+",
        help="Reference KB columns. Defaults to all other *_KB columns.",
    )
    parser.add_argument(
        "--empty-prediction-is-no-relation",
        action="store_true",
        help="Treat blank prediction cells as explicit NO_RELATION instead of skipping them as missing.",
    )
    parser.add_argument(
        "--position-sensitive",
        action="store_true",
        help=(
            "Also calculate position-sensitive relation-sequence micro-F1. "
            "Relations only match when they occur at the same sequence position."
        ),
    )
    parser.add_argument(
        "--argument-order-agnostic",
        action="store_true",
        help=(
            "Also calculate a diagnostic relation-set micro-F1 that treats "
            "the two arguments in each relation as interchangeable."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default="multi_reference_f1_summary.csv",
        help="Path to write the summary CSV. Use '' to disable.",
    )
    parser.add_argument(
        "--details-csv",
        help="Optional per-item details CSV. Only allowed when --prediction-column is set.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.csv)
    fieldnames, rows = load_rows(input_path)
    # The CLI's order-agnostic flag is an additional diagnostic; its directed
    # primary metric must remain compatible with the upstream interface.
    scoring = ScoringConfig(lemmatize=args.lemmatize)
    if scoring.lemmatize and not {"premise", "hypothesis"}.issubset(fieldnames):
        raise SystemExit("--lemmatize requires premise and hypothesis columns for contextual POS.")
    prepare_scoring_resources([scoring])
    available_kb_columns = kb_columns(fieldnames)
    if not available_kb_columns:
        raise SystemExit("No *_KB columns found in the input CSV.")

    if args.prediction_column:
        prediction_columns = [args.prediction_column]
    elif args.reference_columns:
        prediction_columns = generated_llm_kb_columns(fieldnames)
        if not prediction_columns:
            raise SystemExit(
                "No generated LLM prediction columns found. Expected columns matching LLM__*_KB, "
                "or pass --prediction-column explicitly."
            )
    else:
        prediction_columns = available_kb_columns

    details_path = Path(args.details_csv) if args.details_csv else None
    if details_path is not None and len(prediction_columns) != 1:
        raise SystemExit("--details-csv can only be used with --prediction-column.")

    all_results: list[EvaluationResult] = []
    for prediction_column in prediction_columns:
        if prediction_column not in fieldnames:
            raise SystemExit(f"Prediction column not found: {prediction_column}")
        reference_columns = args.reference_columns or [
            column for column in available_kb_columns if column != prediction_column
        ]
        missing_references = [column for column in reference_columns if column not in fieldnames]
        if missing_references:
            raise SystemExit(f"Reference column(s) not found: {', '.join(missing_references)}")
        if not reference_columns:
            raise SystemExit(f"No reference columns available for {prediction_column}.")

        result = evaluate_prediction_column(
            rows,
            prediction_column,
            reference_columns,
            empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
            calculate_position_sensitive=args.position_sensitive,
            calculate_argument_order_agnostic=args.argument_order_agnostic,
            details_path=details_path,
            scoring=scoring,
        )
        all_results.append(result)
        print_result(result)
        print_global_reference_results(
            prediction_column,
            evaluate_global_best_reference(
                rows,
                prediction_column,
                reference_columns,
                empty_prediction_is_no_relation=args.empty_prediction_is_no_relation,
                scoring=scoring,
            ),
        )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        write_summary(summary_path, all_results)
        print(f"\nWrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
