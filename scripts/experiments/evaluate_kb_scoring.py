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
    Counts, ScoringConfig, ScoringContext, is_blank, parse_kb_cell,
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


def evaluate_rows(saved_rows, sources, configs, reference_columns, repeats):
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
        if status == "valid" and not any(references[identifier].values()):
            status = "no_reference"
        statuses[status] += 1
        context = contexts[identifier]
        parsed = parse_kb_cell(raw) if status == "valid" else frozenset()
        for name, config in configs.items():
            key = prompt, model, repeat, name
            acc = accumulators.setdefault(key, {"counts": Counts(), "total_items": 0,
                "evaluated_items": 0, "error_runs": 0, "skipped_missing_prediction": 0,
                "skipped_no_reference": 0, "exact_best_matches": 0, "no_relation_best_matches": 0})
            acc["total_items"] += 1
            row = {"ID": identifier, "prompt": prompt, "model": model, "repeat": repeat,
                   "configuration": name, "status": status, "error": error,
                   "premise": source["premise"], "hypothesis": source["hypothesis"],
                   "raw_KB": raw, "scored_prediction": "", "best_reference_column": "",
                   "raw_reference_KB": "", "scored_reference": "", "tp": "", "fp": "", "fn": "", "item_f1": ""}
            if status != "valid":
                acc["error_runs"] += status == "error"
                acc["skipped_missing_prediction"] += status in {"error", "missing"}
                acc["skipped_no_reference"] += status == "no_reference"
            else:
                prediction = context.kb(parsed, config)
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
            if row["status"] == "valid":
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


def report(summary, configs, statuses, decisions):
    lines = ["# KB scoring diagnostics", "",
        "Saved predictions and annotations are not rewritten. Only their scoring representations change. No filtering, P/H reorientation, underscore replacement, preposition removal or identical-argument suppression is applied.", "",
        f"Slots: {sum(statuses.values()):,}; status counts: {statuses}.", "",
        "Tables report percentages as mean (± sample SD) across generation runs, rounded to one decimal place. Rows are sorted by descending mean micro-F1 before rounding; bold means mark the highest unrounded value in each column. Five generation runs are not five-shot prompting.", "",
        "Micro-precision, micro-recall and micro-F1 use TP, FP and FN aggregated across evaluated problems within each run. Exact match is the percentage of evaluated problems whose entire scored KB equals at least one reference; partial matches do not count.", "",
        "Only the eight evaluated LLMs are included. WordNet and other baseline systems from the paper are not part of this replay.", ""]
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
        "Each condition selects its own best reference by item F1, TP, -FP, -FN, then reference-column name (maximum). P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Zero denominators are undefined. Empty/empty has item selection score 1 and is an exact match, but adds no relation counts. Missing/error slots are excluded identically. Recall is selected-reference recall, not fixed-reference recall.", "",
        "These are KB annotation scores, not LangPro proof-success measurements. Changes diagnose sensitivity to scoring conventions; neither a gain nor semantic correctness is guaranteed.", "",
        "The six deterministic artifacts use UTF-8/LF. The manifest hashes exact output/resource bytes and LF-normalized source/input bytes; local manifests may also include Git metadata. Machine paths, timestamps and environment metadata may differ.", ""]
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
    parser.add_argument("--output", type=Path, help="Optional export to a fresh directory; default: print only.")
    return parser


def run(args):
    configs = configurations(args)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    artifacts = ["configurations.json", "predictions.csv", "metrics_by_run.csv", "summary.csv", "examples.csv", "REPORT.md"]
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
    details, metrics, summary, examples, statuses, decisions = evaluate_rows(saved, sources, configs, args.reference_columns, args.repeats)
    if inputs != {path_label(p): sha256(p) for p in (args.input_csv, args.sample_csv)} or code_hashes != {path_label(p): source_sha256(p) for p in code_paths}:
        raise RuntimeError("Inputs or code changed during replay.")
    text = report(summary, configs, statuses, decisions)
    # Keep redirected output readable on Windows code pages as well as UTF-8.
    printed = text.split("## Evaluation contract")[0].strip()
    print(printed.replace("±", "+/-").replace("↓", "(descending)"))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "configurations.json", {n: asdict(c) for n, c in configs.items()})
        for name, rows, fields in [("predictions.csv", details, None), ("metrics_by_run.csv", metrics, None),
                                  ("summary.csv", summary, None), ("examples.csv", examples, [*details[0], "changed_side"])]:
            write_csv(args.output / name, rows, fields)
        (args.output / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scoring_version": "1.0-symmetric-contextual-pos",
            "git_source": git_source_provenance(code_hashes), "code_sha256": code_hashes,
            "inputs_sha256": inputs, "inputs_lf_sha256": {path_label(p): source_sha256(p) for p in (args.input_csv, args.sample_csv)},
            "outputs_sha256": {n: sha256(args.output / n) for n in artifacts}, "nlp_resources": resources,
            "hash_policy": "Outputs/resources: exact bytes; code and inputs_lf: CRLF converted to LF.",
            "versions": {"python": platform.python_version(), **package_versions(("nltk", "pydantic"))},
            "configurations": {n: asdict(c) for n, c in configs.items()}, "slot_status_counts": statuses,
            "expected_slots": sum(statuses.values()), "metrics_rows": len(metrics), "summary_rows": len(summary),
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
