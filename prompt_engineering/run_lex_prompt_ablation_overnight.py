#!/usr/bin/env python3
"""Run the four-condition LEX prompt ablation and follow-up stability check.

The workflow is resumable:

1. Run four LEX prompt variants on all 223 high-agreement items.
2. Score every prompt/model combination with the existing multi-reference
   scorer and rank prompts by their mean micro-F1 across models.
3. Select the two highest-ranked prompts.
4. Choose a balanced stability sample (default: 100 items).
5. Reuse the primary output as repetition 1 and generate repetitions 2 and 3.
6. Write repeat-level quality summaries and between-run stability metrics.

Rerunning the same command resumes cells that do not yet contain a KB or an
error. Production prompt definitions are not modified. Model relations are
scored without the production KB filtering/lemmatization pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calculate_multi_reference_f1 import parse_kb_cell
from kbprojection.prompts import LASHA_BASE_PROMPT as SOURCE_PROMPT
from prompt_engineering import run_prompt_trial as trial


DEFAULT_OUTPUT_DIR = ROOT / "prompt_engineering" / "lex_prompt_ablation_overnight"
DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-oss-20b",
]

CONSTRAINT_MARKER = "Relation formatting rules:"
EXAMPLE_MARKER = "Examples:"
FINAL_QUERY_MARKER = (
    "Now process the following input while strictly following the above "
    "instructions and formatting."
)

PRECISION_BLOCK = """Additional calibration while preserving all rules above:
- Only output lexical entailment relations that are both factually acceptable and needed to explain the entailment.
- Do not output relations for entailments that follow without a non-trivial lexical bridge.
- Do not use event or social implications as lexical entailment unless the phrase relation is a direct paraphrase.
- Do not output modifier-dropping relations such as old woman -> woman or military men -> men.
- Do not output both an inflected form and a lemmatized form for the same relation.
- If the best relation would violate any existing formatting rule, omit it instead of approximating it.
"""

# The user requested that the original final-query paragraph not be treated as
# part of the prompt variants. The experiment supplies only the item itself
# after the selected LEX prompt sections.
RUNTIME_INPUT = """input:
\tpremise: ${PREMISE}
\thypothesis: ${HYPOTHESIS}
"""


def split_source_prompt() -> tuple[str, str, str]:
    """Return the exact task, constraint, and example sections by deletion."""
    if CONSTRAINT_MARKER not in SOURCE_PROMPT:
        raise ValueError("LEX constraint marker was not found.")
    if EXAMPLE_MARKER not in SOURCE_PROMPT:
        raise ValueError("LEX example marker was not found.")
    if FINAL_QUERY_MARKER not in SOURCE_PROMPT:
        raise ValueError("LEX final-query marker was not found.")

    task_text, after_task = SOURCE_PROMPT.split(CONSTRAINT_MARKER, 1)
    constraint_text, after_constraints = after_task.split(EXAMPLE_MARKER, 1)
    example_text, _removed_final_query = after_constraints.split(FINAL_QUERY_MARKER, 1)

    task = task_text.rstrip()
    constraints = f"{CONSTRAINT_MARKER}{constraint_text}".rstrip()
    examples = f"{EXAMPLE_MARKER}{example_text}".rstrip()
    return task, constraints, examples


def join_sections(*sections: str) -> str:
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


LEX_TASK, LEX_CONSTRAINTS, LEX_EXAMPLES = split_source_prompt()

ABLATION_PROMPTS = [
    trial.CandidatePrompt(
        name="lex_zero_shot_base",
        description=(
            "LEX task section with the constraint and example sections "
            "removed; the item is supplied separately."
        ),
        template=join_sections(LEX_TASK, RUNTIME_INPUT),
    ),
    trial.CandidatePrompt(
        name="lex_zero_shot_constrained",
        description=(
            "LEX task and relation-formatting sections with only the "
            "worked examples and original final-query paragraph removed."
        ),
        template=join_sections(LEX_TASK, LEX_CONSTRAINTS, RUNTIME_INPUT),
    ),
    trial.CandidatePrompt(
        name="lex_few_shot_base",
        description=(
            "LEX task, constraint, and example sections; the original "
            "final-query paragraph is removed and the item is supplied separately."
        ),
        template=join_sections(
            LEX_TASK,
            LEX_CONSTRAINTS,
            LEX_EXAMPLES,
            RUNTIME_INPUT,
        ),
    ),
    trial.CandidatePrompt(
        name="lex_few_shot_precision",
        description=(
            "The few-shot condition plus the precision calibration block."
        ),
        template=join_sections(
            LEX_TASK,
            LEX_CONSTRAINTS,
            LEX_EXAMPLES,
            PRECISION_BLOCK,
            RUNTIME_INPUT,
        ),
    ),
]

PROMPT_NAMES = [prompt.name for prompt in ABLATION_PROMPTS]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header.")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: list[float]) -> float:
    usable = [value for value in values if not math.isnan(value)]
    return statistics.fmean(usable) if usable else float("nan")


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def build_generation_args(
    args: argparse.Namespace,
    *,
    output_csv: Path,
    prompts: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        output_csv=str(output_csv),
        prompts=prompts,
        models=args.models,
        provider=args.provider,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        write_every_jobs=args.write_every_jobs,
        resume=True,
        # Preserve the model's parsed relations. Do not run contextual
        # filtering, lemma generation, swapped-argument generation, or other
        # production KB post-processing.
        no_filter_kb=True,
    )


def write_prompt_document(path: Path) -> None:
    lines = [
        "# LEX prompt ablation",
        "",
        "The original final-query paragraph is excluded from every condition.",
        "The premise and hypothesis are appended through the shared runtime input.",
        "",
    ]
    for prompt in ABLATION_PROMPTS:
        lines.extend(
            [
                f"## {prompt.name}",
                "",
                prompt.description,
                "",
                "```text",
                prompt.template.rstrip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_prompt_leaderboard(
    leaderboard_csv: Path,
    output_csv: Path,
    expected_model_count: int,
) -> list[dict[str, object]]:
    _, rows = read_csv(leaderboard_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("prompt") in PROMPT_NAMES:
            grouped[row["prompt"]].append(row)

    summary: list[dict[str, object]] = []
    for prompt in PROMPT_NAMES:
        prompt_rows = grouped.get(prompt, [])
        if len(prompt_rows) != expected_model_count:
            raise RuntimeError(
                f"{prompt} has {len(prompt_rows)} scored models; "
                f"expected {expected_model_count}."
            )
        summary.append(
            {
                "prompt": prompt,
                "models": len(prompt_rows),
                "mean_micro_f1": fmt(
                    finite_mean([parse_float(row["micro_f1"]) for row in prompt_rows])
                ),
                "mean_precision": fmt(
                    finite_mean([parse_float(row["precision"]) for row in prompt_rows])
                ),
                "mean_recall": fmt(
                    finite_mean([parse_float(row["recall"]) for row in prompt_rows])
                ),
                "mean_exact_best_match_rate": fmt(
                    finite_mean(
                        [parse_float(row["exact_best_match_rate"]) for row in prompt_rows]
                    )
                ),
            }
        )

    summary.sort(key=lambda row: -parse_float(row["mean_micro_f1"]))
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    fields = [
        "rank",
        "prompt",
        "models",
        "mean_micro_f1",
        "mean_precision",
        "mean_recall",
        "mean_exact_best_match_rate",
    ]
    write_csv(output_csv, fields, summary)
    return summary


def selected_prediction_columns(
    prompts: list[str],
    models: list[str],
) -> set[str]:
    selected: set[str] = set()
    for prompt in prompts:
        for model in models:
            columns = trial.generated_columns(prompt, model)
            selected.update([columns.raw_response, columns.kb, columns.error])
    return selected


def ordered_prediction_columns(
    prompts: list[str],
    models: list[str],
) -> list[str]:
    columns: list[str] = []
    for prompt in prompts:
        for model in models:
            generated = trial.generated_columns(prompt, model)
            columns.extend([generated.raw_response, generated.kb, generated.error])
    return columns


def prepare_repeat_resume_file(
    path: Path,
    source_rows: list[dict[str, str]],
    base_fields: list[str],
    selected_prompts: list[str],
    models: list[str],
) -> None:
    """Seed/repair a repeat CSV while preserving already completed cells.

    This prevents primary-run LLM columns from leaking into repeat rows and
    also repairs interrupted files that contain only a header or a subset of
    the expected sample rows.
    """
    generated_fields = ordered_prediction_columns(selected_prompts, models)
    fields = trial.ensure_columns(base_fields, generated_fields)

    existing_by_id: dict[str, dict[str, str]] = {}
    if path.exists():
        _existing_fields, existing_rows = read_csv(path)
        existing_by_id = {
            row.get("ID", ""): row
            for row in existing_rows
            if row.get("ID", "").strip()
        }

    repaired_rows: list[dict[str, object]] = []
    for source in source_rows:
        item_id = source.get("ID", "")
        existing = existing_by_id.get(item_id, {})
        row: dict[str, object] = {
            field: source.get(field, "")
            for field in base_fields
        }
        for field in generated_fields:
            row[field] = existing.get(field, "")
        repaired_rows.append(row)

    write_csv(path, fields, repaired_rows)


def create_repeat_one_from_primary(
    primary_output: Path,
    sample_rows: list[dict[str, str]],
    selected_prompts: list[str],
    models: list[str],
    repeat_one_csv: Path,
) -> None:
    primary_fields, primary_rows = read_csv(primary_output)
    by_id = {row.get("ID", ""): row for row in primary_rows}
    generated = selected_prediction_columns(selected_prompts, models)
    fields = [
        field
        for field in primary_fields
        if not field.startswith("LLM__") or field in generated
    ]
    rows: list[dict[str, object]] = []
    for source in sample_rows:
        item_id = source.get("ID", "")
        if item_id not in by_id:
            raise RuntimeError(f"Stability item {item_id!r} is absent from primary output.")
        primary = by_id[item_id]
        rows.append({field: primary.get(field, "") for field in fields})
    write_csv(repeat_one_csv, fields, rows)


def relation_f1(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    tp = len(left & right)
    fp = len(left - right)
    fn = len(right - left)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 1.0


def calculate_stability(
    repeat_outputs: list[Path],
    selected_prompts: list[str],
    models: list[str],
    output_csv: Path,
) -> None:
    repeat_rows: list[dict[str, dict[str, str]]] = []
    for path in repeat_outputs:
        _, rows = read_csv(path)
        repeat_rows.append({row.get("ID", ""): row for row in rows})

    item_ids = list(repeat_rows[0])
    output_rows: list[dict[str, object]] = []
    for prompt in selected_prompts:
        for model in models:
            columns = trial.generated_columns(prompt, model)
            valid_items = 0
            incomplete_items = 0
            all_identical = 0
            no_relation_flips = 0
            pair_exact: list[float] = []
            pair_f1: list[float] = []

            for item_id in item_ids:
                values: list[frozenset[tuple[str, ...]]] = []
                complete = True
                for rows_by_id in repeat_rows:
                    row = rows_by_id.get(item_id)
                    if (
                        row is None
                        or row.get(columns.error, "").strip()
                        or not row.get(columns.kb, "").strip()
                    ):
                        complete = False
                        break
                    values.append(parse_kb_cell(row[columns.kb]))
                if not complete:
                    incomplete_items += 1
                    continue

                valid_items += 1
                if len(set(values)) == 1:
                    all_identical += 1
                if any(not value for value in values) and any(value for value in values):
                    no_relation_flips += 1
                for left, right in itertools.combinations(values, 2):
                    pair_exact.append(float(left == right))
                    pair_f1.append(relation_f1(left, right))

            output_rows.append(
                {
                    "prompt": prompt,
                    "model": trial.model_slug(model),
                    "requested_repeats": len(repeat_outputs),
                    "valid_items": valid_items,
                    "incomplete_items": incomplete_items,
                    "all_runs_identical_rate": fmt(
                        all_identical / valid_items if valid_items else float("nan")
                    ),
                    "mean_pairwise_exact": fmt(finite_mean(pair_exact)),
                    "mean_pairwise_kb_f1": fmt(finite_mean(pair_f1)),
                    "no_relation_flip_rate": fmt(
                        no_relation_flips / valid_items if valid_items else float("nan")
                    ),
                }
            )

    fields = [
        "prompt",
        "model",
        "requested_repeats",
        "valid_items",
        "incomplete_items",
        "all_runs_identical_rate",
        "mean_pairwise_exact",
        "mean_pairwise_kb_f1",
        "no_relation_flip_rate",
    ]
    write_csv(output_csv, fields, output_rows)


def summarize_quality_across_repeats(
    repeat_leaderboards: list[Path],
    selected_prompts: list[str],
    output_csv: Path,
) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for leaderboard in repeat_leaderboards:
        _, rows = read_csv(leaderboard)
        for row in rows:
            if row.get("prompt") in selected_prompts:
                grouped[(row["prompt"], row["model"])].append(
                    parse_float(row["micro_f1"])
                )

    output_rows: list[dict[str, object]] = []
    for (prompt, model), values in sorted(grouped.items()):
        usable = [value for value in values if not math.isnan(value)]
        output_rows.append(
            {
                "prompt": prompt,
                "model": model,
                "runs": len(usable),
                "mean_micro_f1": fmt(finite_mean(usable)),
                "sample_sd_micro_f1": fmt(
                    statistics.stdev(usable) if len(usable) > 1 else float("nan")
                ),
                "min_micro_f1": fmt(min(usable) if usable else float("nan")),
                "max_micro_f1": fmt(max(usable) if usable else float("nan")),
            }
        )
    fields = [
        "prompt",
        "model",
        "runs",
        "mean_micro_f1",
        "sample_sd_micro_f1",
        "min_micro_f1",
        "max_micro_f1",
    ]
    write_csv(output_csv, fields, output_rows)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    selected_prompts: list[str] | None,
) -> None:
    selected = (
        "\n".join(f"- `{prompt}`" for prompt in selected_prompts)
        if selected_prompts
        else "_Selected after the primary run._"
    )
    path.write_text(
        f"""# LEX prompt ablation overnight experiment

## Design

- Primary data: all high-agreement items
- Agreement source: {args.agreement_csv}
- Derived unanimous subset: {args.agreed_subset_csv}
- Primary prompts: {len(PROMPT_NAMES)}
- Models: {len(args.models)}
- Stability sample: {args.stability_sample_size} items
- Stability repetitions: {args.repeats}
- KB filtering/lemmatization pipeline enabled: no
- Relation source: directly parsed model response
- Temperature explicitly supplied: no

## Selected prompts

{selected}

## Resume behavior

Rerun the same command. Completed KB or error cells are skipped.

## Main artifacts

- `prompts.md`: exact tested prompt templates
- `primary_outputs.csv`: raw responses, parsed KBs, and errors
- `primary_leaderboard.csv`: model-level quality scores
- `primary_prompt_summary.csv`: prompt means used for selection
- `stability_sample.csv`: balanced follow-up sample
- `repeat_{{1,2,3}}_outputs.csv`: repeated outputs
- `quality_across_repeats.csv`: repeat-level quality summary
- `stability_metrics.csv`: agreement between repeated outputs
""",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agreement-csv",
        default=str(
            ROOT / "data" / "annotator_agreement" / "iaa_overview_edit.csv"
        ),
        help="Edited IAA overview used to derive the 223 high-agreement items.",
    )
    parser.add_argument(
        "--agreed-subset-csv",
        help=(
            "Optional destination for the derived high-agreement subset. "
            "Defaults to agreed_subset.csv in --output-dir."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--reference-columns",
        nargs="+",
        default=trial.DEFAULT_REFERENCE_COLUMNS,
    )
    parser.add_argument("--stability-sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--write-every-jobs", type=int, default=10)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write prompts and samples without making API calls.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agreed_subset_csv = (
        Path(args.agreed_subset_csv)
        if args.agreed_subset_csv
        else output_dir / "agreed_subset.csv"
    )
    args.agreed_subset_csv = str(agreed_subset_csv)

    # run_generation resolves candidates through this module-level registry.
    # The replacement is local to this process and does not edit production code.
    trial.CANDIDATE_PROMPTS = ABLATION_PROMPTS

    base_fields, full_rows = trial.build_agreed_subset(
        Path(args.agreement_csv),
        agreed_subset_csv,
    )
    if not full_rows:
        raise RuntimeError("The high-agreement subset is empty.")

    primary_sample = output_dir / "primary_sample.csv"
    trial.write_rows(primary_sample, base_fields, full_rows)
    write_prompt_document(output_dir / "prompts.md")

    stability_rows = trial.choose_dev_sample(
        full_rows,
        args.reference_columns,
        min(args.stability_sample_size, len(full_rows)),
        0.5,
    )
    stability_sample = output_dir / "stability_sample.csv"
    trial.write_rows(stability_sample, base_fields, stability_rows)
    write_readme(output_dir / "README.md", args, selected_prompts=None)

    primary_jobs = len(full_rows) * len(PROMPT_NAMES) * len(args.models)
    later_jobs = (
        len(stability_rows)
        * 2
        * len(args.models)
        * max(0, args.repeats - 1)
    )
    print(f"Primary grid: {primary_jobs} calls.", flush=True)
    print(f"Later stability calls after selection: {later_jobs}.", flush=True)
    print(f"Maximum new calls for a fresh run: {primary_jobs + later_jobs}.", flush=True)

    if args.prepare_only:
        print(f"Prepared experiment in {output_dir}; no model calls were made.")
        return

    primary_output = output_dir / "primary_outputs.csv"
    await trial.run_generation(
        build_generation_args(
            args,
            output_csv=primary_output,
            prompts=PROMPT_NAMES,
        ),
        full_rows,
        base_fields,
    )

    primary_summary = output_dir / "primary_score_summary.csv"
    primary_leaderboard = output_dir / "primary_leaderboard.csv"
    trial.score_outputs(
        primary_output,
        primary_summary,
        args.reference_columns,
        empty_prediction_is_no_relation=False,
        metadata_csv=primary_leaderboard,
    )
    prompt_summary = summarize_prompt_leaderboard(
        primary_leaderboard,
        output_dir / "primary_prompt_summary.csv",
        len(args.models),
    )
    selected_prompts = [str(row["prompt"]) for row in prompt_summary[:2]]
    print(f"Selected top prompts: {', '.join(selected_prompts)}", flush=True)
    write_readme(output_dir / "README.md", args, selected_prompts)

    # Reload pristine sample rows. The primary generator mutates its input row
    # dictionaries by adding all four prompts' output columns; those columns
    # must not be mistaken for completed repeated-run jobs.
    stability_fields, stability_rows = read_csv(stability_sample)

    repeat_outputs: list[Path] = []
    repeat_leaderboards: list[Path] = []

    repeat_one_output = output_dir / "repeat_1_outputs.csv"
    create_repeat_one_from_primary(
        primary_output,
        stability_rows,
        selected_prompts,
        args.models,
        repeat_one_output,
    )
    repeat_outputs.append(repeat_one_output)

    for repeat in range(1, args.repeats + 1):
        output_csv = output_dir / f"repeat_{repeat}_outputs.csv"
        if repeat >= 2:
            prepare_repeat_resume_file(
                output_csv,
                stability_rows,
                stability_fields,
                selected_prompts,
                args.models,
            )
            await trial.run_generation(
                build_generation_args(
                    args,
                    output_csv=output_csv,
                    prompts=selected_prompts,
                ),
                stability_rows,
                stability_fields,
            )
        leaderboard = output_dir / f"repeat_{repeat}_leaderboard.csv"
        trial.score_outputs(
            output_csv,
            output_dir / f"repeat_{repeat}_score_summary.csv",
            args.reference_columns,
            empty_prediction_is_no_relation=False,
            metadata_csv=leaderboard,
        )
        repeat_leaderboards.append(leaderboard)
        if repeat >= 2:
            repeat_outputs.append(output_csv)

    summarize_quality_across_repeats(
        repeat_leaderboards,
        selected_prompts,
        output_dir / "quality_across_repeats.csv",
    )
    calculate_stability(
        repeat_outputs,
        selected_prompts,
        args.models,
        output_dir / "stability_metrics.csv",
    )

    print(f"Experiment complete. Results: {output_dir}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.stability_sample_size < 2:
        raise SystemExit("--stability-sample-size must be at least 2.")
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if args.max_retries < 0:
        raise SystemExit("--max-retries cannot be negative.")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be greater than 0.")
    if args.write_every_jobs < 1:
        raise SystemExit("--write-every-jobs must be at least 1.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
