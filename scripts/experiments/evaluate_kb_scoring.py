#!/usr/bin/env python3
"""Diagnose KB F1 with symmetric scoring rules, without prediction filtering."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calculate_multi_reference_f1 import (
    Counts, NON_ENTAILMENT_POLICIES, NON_ENTAILMENT_WRONG_RELATION,
    ScoringConfig, ScoringContext, is_blank, parse_kb_cell, model_prediction_exclusion,
    prepare_scoring_resources,
)
from scripts.experiments.replay_utils import (
    REFERENCE_COLUMNS, choose_reference, git_source_provenance, path_label,
    prepare_slots, read_csv, resource_fingerprints, sha256, source_sha256,
    summarize, write_csv, write_json,
)

LABELS = {"original": "Original", "lemmas": "Lemmas (both sides)",
          "argument_order_agnostic": "Argument order agnostic", "combined": "Lemmas + argument order agnostic"}
MODEL_LABELS = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google/gemma-3-4b-it": "Gemma 3 4B",
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
}

NORMALIZATION_CONFIGURATIONS = ("original", "lemmas", "argument_order_agnostic", "combined")
SCORED_STATUSES = {"valid", "non_entailment_wrong"}


def exact_match(row):
    """Whether the selected reference exactly matches this scoring representation."""
    return row["status"] in SCORED_STATUSES and row["scored_prediction"] == row["scored_reference"]


def normalization_trace(details):
    """Join four scoring variants into one audit row per saved model answer.

    The regular ``predictions.csv`` deliberately retains one row per scoring
    variant.  This companion view is for reviewing which individual answers
    gain (or lose) item-level F1 after a normalization; it includes excluded
    slots as well, so its row count remains the full prediction total.
    """
    expected = set(NORMALIZATION_CONFIGURATIONS)
    grouped = {}
    for row in details:
        key = tuple(row[name] for name in ("ID", "prompt", "model", "repeat"))
        variants = grouped.setdefault(key, {})
        if row["configuration"] in variants:
            raise ValueError(f"Duplicate scoring variant for prediction slot: {key}")
        variants[row["configuration"]] = row
    if any(set(variants) != expected for variants in grouped.values()):
        raise ValueError("The normalization trace requires original, lemma, unordered, and combined scores.")

    result = []
    for key, variants in sorted(grouped.items()):
        original = variants["original"]
        row = {name: original[name] for name in ("ID", "prompt", "model", "repeat", "status", "error",
                                                   "premise", "hypothesis", "raw_KB")}
        if original["status"] in SCORED_STATUSES:
            scores = {name: float(variants[name]["item_f1"]) for name in NORMALIZATION_CONFIGURATIONS}
            row.update({
                "original_item_f1": scores["original"],
                "lemmatized_item_f1": scores["lemmas"],
                "unordered_item_f1": scores["argument_order_agnostic"],
                "combined_item_f1": scores["combined"],
                "lemma_item_f1_delta": scores["lemmas"] - scores["original"],
                "unordered_item_f1_delta": scores["argument_order_agnostic"] - scores["original"],
                "combined_item_f1_delta": scores["combined"] - scores["original"],
            })
            best_score = max(scores.values())
            row["best_scoring_modes"] = ";".join(name for name in NORMALIZATION_CONFIGURATIONS
                                                    if scores[name] == best_score)
            row["best_item_f1_gain_over_original"] = best_score - scores["original"]
            row["any_item_f1_gain"] = int(best_score > scores["original"])
            for name, label in (("lemmas", "lemma"),
                                ("argument_order_agnostic", "unordered"),
                                ("combined", "combined")):
                row[f"{label}_item_f1_gain"] = int(scores[name] > scores["original"])
                row[f"{label}_item_f1_loss"] = int(scores[name] < scores["original"])
        else:
            for field in ("original_item_f1", "lemmatized_item_f1", "unordered_item_f1", "combined_item_f1",
                          "lemma_item_f1_delta", "unordered_item_f1_delta", "combined_item_f1_delta",
                          "best_scoring_modes", "best_item_f1_gain_over_original", "any_item_f1_gain",
                          "lemma_item_f1_gain", "lemma_item_f1_loss", "unordered_item_f1_gain",
                          "unordered_item_f1_loss", "combined_item_f1_gain", "combined_item_f1_loss"):
                row[field] = ""
        for name, label in (("original", "original"), ("lemmas", "lemmatized"),
                            ("argument_order_agnostic", "unordered"), ("combined", "combined")):
            variant = variants[name]
            row.update({
                f"{label}_exact_match": int(exact_match(variant)) if original["status"] in SCORED_STATUSES else "",
                f"{label}_best_reference_column": variant["best_reference_column"],
                f"{label}_scored_prediction": variant["scored_prediction"],
                f"{label}_scored_reference": variant["scored_reference"],
                f"{label}_tp": variant["tp"], f"{label}_fp": variant["fp"], f"{label}_fn": variant["fn"],
            })
        if original["status"] in SCORED_STATUSES:
            raw_exact = exact_match(variants["original"])
            for name, label in (("lemmas", "lemma"),
                                ("argument_order_agnostic", "unordered"),
                                ("combined", "combined")):
                row[f"{label}_new_exact_match"] = int(not raw_exact and exact_match(variants[name]))
        else:
            for label in ("lemma", "unordered", "combined"):
                row[f"{label}_new_exact_match"] = ""
        result.append(row)
    return result


def normalization_summary(trace):
    """Count answer-level F1 changes, by model, for an appendix-friendly overview."""
    rows_by_model = defaultdict(list)
    for row in trace:
        rows_by_model[row["model"]].append(row)
    summary = []
    for model, rows in sorted(rows_by_model.items()):
        valid = [row for row in rows if row["status"] in SCORED_STATUSES]
        item = {"prompt": rows[0]["prompt"], "model": model, "total_prediction_slots": len(rows),
                "retained_prediction_slots": len(valid), "excluded_prediction_slots": len(rows) - len(valid)}
        for label in ("lemma", "unordered", "combined"):
            for change in ("gain", "loss"):
                count = sum(row[f"{label}_item_f1_{change}"] for row in valid)
                item[f"{label}_item_f1_{change}_count"] = count
                item[f"{label}_item_f1_{change}_percent_of_retained"] = count / len(valid) if valid else math.nan
            exact = sum(row[f"{label}_new_exact_match"] for row in valid)
            item[f"{label}_new_exact_match_count"] = exact
            item[f"{label}_new_exact_match_percent_of_retained"] = exact / len(valid) if valid else math.nan
        summary.append(item)
    return summary


def configurations(args):
    if args.compare:
        if args.lemmatize or args.argument_order_agnostic:
            raise ValueError("Use --compare or individual scoring flags, not both.")
        return dict(zip(LABELS, (ScoringConfig(), ScoringConfig(True),
                                ScoringConfig(False, True), ScoringConfig(True, True))))
    config = ScoringConfig(args.lemmatize, args.argument_order_agnostic)
    name = "combined" if all(asdict(config).values()) else "lemmas" if config.lemmatize else "argument_order_agnostic" if config.argument_order_agnostic else "original"
    return {name: config}


def display(kb):
    # JSON arrays preserve empty fields and legacy non-binary tuples unambiguously.
    return json.dumps(sorted(kb), ensure_ascii=False)


def evaluate_rows(saved_rows, sources, configs, reference_columns, repeats,
                  non_entailment_policy="exclude"):
    if non_entailment_policy not in NON_ENTAILMENT_POLICIES:
        raise ValueError(f"Unknown non-entailment policy: {non_entailment_policy}")
    slots = prepare_slots(saved_rows, sources, repeats)
    contexts, references = {}, {}
    for source in sources:
        identifier = source["ID"]
        context = contexts[identifier] = ScoringContext(source["premise"], source["hypothesis"])
        raw_refs = [(c, parse_kb_cell(source[c])) for c in reference_columns if not is_blank(source.get(c, ""))]
        references[identifier] = {name: [(c, context.kb(kb, config)) for c, kb in raw_refs]
                                  for name, config in configs.items()}
    details, examples, accumulators = [], [], {}
    statuses, example_counts, lemma_decisions = Counter(), Counter(), Counter()
    # Unique argument decisions per problem, rather than five repeated generations.
    for prompt, model, repeat, source, saved in slots:
        identifier = source["ID"]
        raw = saved.get("KB", "") if saved else ""
        error = saved.get("error", "").strip() if saved else "missing saved-output row"
        status = "error" if saved and error else "missing" if is_blank(raw) else "valid"
        if status == "valid" and model_prediction_exclusion(saved, "KB") == "non_entailment":
            status = ("non_entailment_wrong" if non_entailment_policy == "always-wrong"
                      else "non_entailment")
        if status in {"valid", "non_entailment_wrong"} and not any(references[identifier].values()):
            status = "no_reference"
        statuses[status] += 1
        context = contexts[identifier]
        parsed = (frozenset({NON_ENTAILMENT_WRONG_RELATION})
                  if status == "non_entailment_wrong"
                  else parse_kb_cell(raw) if status == "valid" else frozenset())
        for name, config in configs.items():
            key = prompt, model, repeat, name
            acc = accumulators.setdefault(key, {"counts": Counts(), "total_items": 0,
                "evaluated_items": 0, "error_runs": 0, "skipped_missing_prediction": 0,
                "skipped_non_entailment": 0, "scored_non_entailment_as_wrong": 0,
                "skipped_no_reference": 0, "exact_best_matches": 0, "no_relation_best_matches": 0})
            acc["total_items"] += 1
            row = {"ID": identifier, "prompt": prompt, "model": model, "repeat": repeat,
                   "configuration": name, "status": status, "error": error,
                   "premise": source["premise"], "hypothesis": source["hypothesis"],
                   "raw_KB": raw, "scored_prediction": "", "best_reference_column": "",
                   "raw_reference_KB": "", "scored_reference": "", "tp": "", "fp": "", "fn": "", "item_f1": ""}
            if status not in {"valid", "non_entailment_wrong"}:
                acc["error_runs"] += status == "error"
                acc["skipped_missing_prediction"] += status in {"error", "missing"}
                acc["skipped_non_entailment"] += status == "non_entailment"
                acc["skipped_no_reference"] += status == "no_reference"
            else:
                prediction = context.kb(parsed, config)
                acc["scored_non_entailment_as_wrong"] += status == "non_entailment_wrong"
                score, column, reference, counts = choose_reference(prediction, references[identifier][name])
                acc["counts"].add(counts)
                acc["evaluated_items"] += 1
                acc["exact_best_matches"] += prediction == reference
                acc["no_relation_best_matches"] += prediction == reference and not prediction
                row.update(scored_prediction=display(prediction), best_reference_column=column,
                           raw_reference_KB=source[column], scored_reference=display(reference),
                           tp=counts.tp, fp=counts.fp, fn=counts.fn, item_f1=score)
                for side, original, scored in (("prediction", parsed, prediction),
                                               ("reference", parse_kb_cell(source[column]), reference)):
                    if original != scored and example_counts[name, side] < 5:
                        examples.append({**row, "changed_side": side})
                        example_counts[name, side] += 1
            details.append(row)
    # Collect cached argument decisions once for each problem, across both sides.
    if any(c.lemmatize for c in configs.values()):
        arguments = defaultdict(set)
        for row in details:
            if row["status"] in SCORED_STATUSES:
                for relation in parse_kb_cell(row["raw_KB"]):
                    arguments[row["ID"]].update(relation)
        for source in sources:
            for column in reference_columns:
                for relation in parse_kb_cell(source.get(column, "")):
                    arguments[source["ID"]].update(relation)
        for identifier, values in arguments.items():
            lemma_decisions.update(contexts[identifier].argument(a)[1] for a in values)
    metrics = []
    for (prompt, model, repeat, name), acc in sorted(accumulators.items()):
        counts = acc.pop("counts")
        metrics.append({"prompt": prompt, "model": model, "repeat": repeat, "configuration": name,
            "reference_columns": ";".join(reference_columns), **acc,
            "tp": counts.tp, "fp": counts.fp, "fn": counts.fn,
            "precision": counts.precision, "recall": counts.recall, "micro_f1": counts.f1,
            "exact_best_match_rate": acc["exact_best_matches"] / acc["evaluated_items"] if acc["evaluated_items"] else math.nan})
    return details, metrics, summarize(metrics), examples, dict(sorted(statuses.items())), dict(sorted(lemma_decisions.items()))


def normalization_summary_latex(summary):
    """Render a compact appendix table; detailed evidence stays in the CSV trace."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Retained} & \textbf{Lemma F1 $\uparrow$} & \textbf{Unordered F1 $\uparrow$} & \textbf{Combined F1 $\uparrow$} \\",
        r"\midrule",
    ]
    for row in summary:
        def gain(label):
            return (f"{row[f'{label}_item_f1_gain_count']:,} "
                    f"({row[f'{label}_item_f1_gain_percent_of_retained'] * 100:.1f}\%)")
        lines.append(
            f"{MODEL_LABELS.get(row['model'], row['model'])} & "
            f"{row['retained_prediction_slots']:,}/{row['total_prediction_slots']:,} & "
            f"{gain('lemma')} & {gain('unordered')} & {gain('combined')} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Answer-level effects of the alternative LEX matching rules across five runs. "
        r"Each entry gives the number (and percentage of retained predictions) whose individual F1 is higher than under raw matching. "
        r"The same prediction may appear in multiple columns; counts are therefore not additive. "
        r"Excluded errors and final non-entailment answers are included in neither numerator nor denominator. "
        r"The accompanying \texttt{normalization\_trace.csv} gives the before/after score, selected reference, and scored relation sets for every prediction slot.}",
        r"\label{tab:lex-normalization-trace}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def report(summary, configs, statuses, decisions, trace_summary=None):
    lines = ["# KB scoring diagnostics", "",
        "Saved predictions and annotations are not rewritten. Only their scoring representations change. No filtering, P/H reorientation, underscore replacement, preposition removal or identical-argument suppression is applied.", "",
        f"Slots: {sum(statuses.values()):,}; status counts: {statuses}.", "",
        "Tables report percentages as mean (± sample SD) across generation runs, rounded to one decimal place. Rows are sorted by descending mean micro-F1 before rounding; bold means mark the highest unrounded value in each column. Five generation runs are not five-shot prompting.", "",
        "Micro-precision, micro-recall and micro-F1 use TP, FP and FN aggregated across evaluated problems within each run. Exact match is the percentage of evaluated problems whose entire scored KB equals at least one reference; partial matches do not count.", "",
        "Only the eight evaluated LLMs are included. WordNet and other baseline systems from the paper are not part of this replay.", ""]
    if trace_summary is not None:
        lines += ["## Per-answer normalization trace", "",
                  "`normalization_trace.csv` has exactly one row for every saved prediction slot. It places original, lemma, unordered, and combined item-level F1 values side by side, together with their score deltas, exact-match flags, selected references, and scored relation sets.", "",
                  "`normalization_gains.csv` is the subset with a higher item-level F1 under at least one normalized condition. `normalization_summary.csv` and `normalization_summary.tex` count those gains per model. A gain identifies a score change under the stated matching rule; it does not by itself establish that a generated relation is semantically correct.", ""]
    groups = defaultdict(list)
    for row in summary:
        groups[row["configuration"], row["prompt"]].append(row)
    metrics = ("exact_best_match_rate", "micro_f1", "precision", "recall")
    for name in configs:
        for (configuration, prompt), rows in sorted(groups.items()):
            if configuration != name:
                continue
            lines += [f"## {LABELS[name]}", "", f"Prompt: `{prompt}`.", "",
                      "| Model | Exact match | Micro-F1 ↓ | Micro-Precision | Micro-Recall |",
                      "|---|---:|---:|---:|---:|"]
            best = {metric: max(r['mean_' + metric] for r in rows) for metric in metrics}
            for row in sorted(rows, key=lambda r: (-r['mean_micro_f1'], r['model'])):
                cells = []
                for metric in metrics:
                    mean = row['mean_' + metric]
                    value = f"{mean * 100:.1f}"
                    if mean == best[metric]:
                        value = f"**{value}**"
                    cells.append(f"{value} (± {row['sample_stddev_' + metric] * 100:.1f})")
                model = MODEL_LABELS.get(row['model'], row['model'])
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["", "## Evaluation contract", "",
        "Original uses the original directed relation-set parser: case, surrounding whitespace and question marks are normalized; predicate names, duplicates and relation-list order are ignored. Extra-comma tuples and empty arguments retain legacy acceptance. No new syntax rejection is added.", "",
        "Lemma scoring applies the SAME contextual POS rule to every argument of predictions and every reference. Exact contiguous spans are searched in both premise and hypothesis, independent of argument position. All occurrences must agree on the WordNet POS sequence (noun/verb/adjective/adverb); otherwise the original argument is retained. Missing spans also remain unchanged. POS comes only from P/H, never from a gold relation choice. This is conservative contextual lemma comparison, not exhaustive equivalence of all inflections or proof of semantic equivalence.", "",
        f"Lemma decisions for unique (problem, argument) values across valid predictions and all references: {decisions}.", "",
        "Argument-order-agnostic scoring sorts the two arguments of binary tuples on BOTH sides. Other tuple arities remain directed. In the combined condition, lemmatization precedes sorting. Canonicalization may merge duplicates, changing denominators; self-pairs remain present.", "",
        "Each condition selects its own best reference by item F1, TP, -FP, -FN, then reference-column name (maximum). P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Zero denominators are undefined. Empty/empty has item selection score 1 and is an exact match, but adds no relation counts. Missing/error slots and final non-entailment answers are excluded identically, counted separately. F1 is conditional on retained outputs. Recall is selected-reference recall, not fixed-reference recall.", "",
        "These are KB annotation scores, not LangPro proof-success measurements. Changes diagnose sensitivity to scoring conventions; neither a gain nor semantic correctness is guaranteed.", "",
        "The deterministic exports use UTF-8/LF. The manifest hashes exact output/resource bytes and LF-normalized source/input bytes; local manifests may also include Git metadata. Machine paths, timestamps and environment metadata may differ.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Example: uv run python scripts/experiments/evaluate_kb_scoring.py --compare")
    parser.add_argument("--compare", action="store_true", help="Original, lemmas on both sides, argument order agnostic, and combined.")
    parser.add_argument("--lemmatize", action="store_true", help="Use contextual POS lemmas on BOTH predictions and references.")
    parser.add_argument("--argument-order-agnostic", action="store_true", help="Score binary (a,b)/(b,a) as equivalent on BOTH sides.")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_outputs.csv")
    parser.add_argument("--sample-csv", type=Path, default=ROOT / "data/all_usable_items_362.csv")
    parser.add_argument("--reference-columns", nargs="+", default=REFERENCE_COLUMNS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--non-entailment-policy", choices=NON_ENTAILMENT_POLICIES,
        default="exclude",
        help="Exclude final non-entailment answers or score each as a guaranteed mismatch.",
    )
    parser.add_argument("--output", type=Path, help="Optional export to a fresh directory; default: print only.")
    return parser


def run(args):
    configs = configurations(args)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    trace_requested = set(configs) == set(NORMALIZATION_CONFIGURATIONS)
    artifacts = ["configurations.json", "predictions.csv", "metrics_by_run.csv", "summary.csv", "examples.csv", "REPORT.md"]
    if trace_requested:
        artifacts += ["normalization_trace.csv", "normalization_gains.csv", "normalization_summary.csv", "normalization_summary.tex"]
    if args.output and any((args.output / n).exists() for n in [*artifacts, "manifest.json"]):
        raise ValueError("Output already contains replay artifacts; choose a fresh directory.")
    inputs = {path_label(p): sha256(p) for p in (args.input_csv, args.sample_csv)}
    code_paths = [Path(__file__), ROOT / "calculate_multi_reference_f1.py", ROOT / "calculate_inter_annotator_agreement.py",
                  ROOT / "scripts/experiments/replay_utils.py",
                  *sorted((ROOT / "kbprojection").rglob("*.py"))]
    code_hashes = {path_label(p): source_sha256(p) for p in code_paths}
    fields, sources = read_csv(args.sample_csv)
    required = {"ID", "premise", "hypothesis", *args.reference_columns}
    if not required.issubset(fields):
        raise ValueError(f"Missing sample columns: {sorted(required - set(fields))}")
    _, saved = read_csv(args.input_csv)
    resources = resource_fingerprints(prepare_scoring_resources(configs.values()))
    details, metrics, summary, examples, statuses, decisions = evaluate_rows(
        saved, sources, configs, args.reference_columns, args.repeats,
        args.non_entailment_policy,
    )
    if inputs != {path_label(p): sha256(p) for p in (args.input_csv, args.sample_csv)} or code_hashes != {path_label(p): source_sha256(p) for p in code_paths}:
        raise RuntimeError("Inputs or code changed during replay.")
    trace = normalization_trace(details) if trace_requested else None
    trace_summary = normalization_summary(trace) if trace_requested else None
    text = report(summary, configs, statuses, decisions, trace_summary)
    # Keep redirected output readable on Windows code pages as well as UTF-8.
    printed = text.split("## Evaluation contract")[0].strip()
    print(printed.replace("±", "+/-").replace("↓", "(descending)"))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "configurations.json", {n: asdict(c) for n, c in configs.items()})
        for name, rows, fields in [("predictions.csv", details, None), ("metrics_by_run.csv", metrics, None),
                                  ("summary.csv", summary, None), ("examples.csv", examples, [*details[0], "changed_side"])]:
            write_csv(args.output / name, rows, fields)
        if trace_requested:
            write_csv(args.output / "normalization_trace.csv", trace)
            write_csv(args.output / "normalization_gains.csv", [row for row in trace if row["any_item_f1_gain"] == 1],
                      list(trace[0]))
            write_csv(args.output / "normalization_summary.csv", trace_summary)
            (args.output / "normalization_summary.tex").write_text(
                normalization_summary_latex(trace_summary), encoding="utf-8", newline="\n")
        (args.output / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scoring_version": "1.0-symmetric-contextual-pos",
            "git_source": git_source_provenance(code_hashes), "code_sha256": code_hashes,
            "inputs_sha256": inputs, "inputs_lf_sha256": {path_label(p): source_sha256(p) for p in (args.input_csv, args.sample_csv)},
            "outputs_sha256": {n: sha256(args.output / n) for n in artifacts}, "nlp_resources": resources,
            "hash_policy": "Outputs/resources: exact bytes; code and inputs_lf: CRLF converted to LF.",
            "versions": {"python": platform.python_version(), **package_versions(("nltk", "pydantic"))},
            "configurations": {n: asdict(c) for n, c in configs.items()},
            "non_entailment_policy": args.non_entailment_policy, "slot_status_counts": statuses,
            "expected_slots": sum(statuses.values()), "metrics_rows": len(metrics), "summary_rows": len(summary),
            "normalization_trace_rows": len(trace) if trace is not None else 0,
            "normalization_gain_rows": sum(row["any_item_f1_gain"] == 1 for row in trace) if trace is not None else 0,
            "lemma_decisions_unique_problem_argument": decisions,
            "stored_predictions_modified": False, "stored_annotations_modified": False,
            "scoring_representation_normalized_on_both_sides": True,
            "llm_calls": 0, "langpro_calls": 0, "download_calls": 0}
        write_json(args.output / "manifest.json", manifest)
        print(f"\nSaved artifacts to {args.output}")
    return statuses


def package_versions(packages):
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as a distribution"
    return versions


if __name__ == "__main__":
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (ValueError, LookupError) as exc:
        parser.exit(2, f"Error: {exc}\n")
