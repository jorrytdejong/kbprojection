#!/usr/bin/env python3
"""Run LEX/Stefan prompts on SNLI-1K/curated-365, with and without WordNet.

Each child run checkpoints one JSONL record per problem. Re-running this script
resumes matching outputs and skips records already safely written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNLI = ROOT / "data" / "snli_train_entailment_1k.jsonl"
CURATED_DEFAULT = ROOT.parent / "drive-ready-updated-20260913T173351Z" / "population_curated365.jsonl"
OUT_DIR = ROOT / "results" / "prompt_matrix_gemini_3_1_flash_lite"
WORDNET_MODES = (False, True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("id", row.get("problem_id", ""))) for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} must contain unique non-empty problem ids")
    return rows


def summarize(path: Path, prompt: str, dataset: str, wordnet: bool) -> dict[str, Any]:
    rows = read_jsonl(path)
    baseline = one_shot = final = 0
    for row in rows:
        outcome = row.get("outcome", {})
        gold = row.get("problem", {}).get("gold_label", outcome.get("gold_label"))
        baseline_correct = outcome.get("pred_baseline") == gold
        one_shot_correct = baseline_correct
        for step in row.get("timeline", []):
            if step.get("phase") == "langpro_with_kb" and step.get("attempt") == 1:
                one_shot_correct = step.get("output", {}).get("matches_gold") is True
                break
        baseline += int(baseline_correct)
        one_shot += int(one_shot_correct)
        final += int(outcome.get("solved") is True)
    n = len(rows)
    return {
        "prompt": prompt,
        "dataset": dataset,
        "wordnet": wordnet,
        "problems": n,
        "baseline_solved": baseline,
        "one_shot_solved": one_shot,
        "agentic_final_solved": final,
        "baseline_rate": round(baseline / n, 4) if n else None,
        "one_shot_rate": round(one_shot / n, 4) if n else None,
        "agentic_final_rate": round(final / n, 4) if n else None,
        "jsonl": str(path),
    }


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "prompt", "dataset", "wordnet", "problems", "baseline_solved",
        "one_shot_solved", "agentic_final_solved", "baseline_rate",
        "one_shot_rate", "agentic_final_rate", "jsonl",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--curated-input", type=Path, default=CURATED_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--prompts", nargs="+", choices=("lex", "stefan"), default=("lex", "stefan"),
        help="Prompt profiles to run (in order). Use --prompts stefan for the July 13 prompt only.",
    )
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Problems per child run at once; 1 is safest for LangPro API/cache.")
    parser.add_argument(
        "--retry-baseline-prover-errors",
        action="store_true",
        help="On resume, retry checkpoint rows whose baseline LangPro request failed; preserve backups.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned commands without calling APIs.")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    datasets = (("snli_1k", SNLI), ("curated_365", args.curated_input.resolve()))
    for name, path in datasets:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")
        rows = read_jsonl(path)
        if name == "snli_1k" and len(rows) != 1000:
            raise ValueError(f"Expected 1000 SNLI rows, found {len(rows)} in {path}")
        if name == "curated_365" and len(rows) != 365:
            raise ValueError(f"Expected 365 curated rows, found {len(rows)} in {path}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    matrix = [
        (prompt, dataset_name, input_path)
        for prompt in args.prompts
        for dataset_name, input_path in (("snli_1k", SNLI), ("curated_365", datasets[1][1]))
    ]
    total = len(matrix) * len(WORDNET_MODES)
    job_num = 0

    for prompt, dataset_name, input_path in matrix:
        for wordnet in WORDNET_MODES:
            job_num += 1
            suffix = "wordnet_on" if wordnet else "wordnet_off"
            output = args.output_dir / f"{dataset_name}_{prompt}_{suffix}.jsonl"
            log_path = args.output_dir / f"{dataset_name}_{prompt}_{suffix}.log"
            command = [
                sys.executable, "-m", "agentic_pipeline_langpro.cli",
                "--input", str(input_path), "--output", str(output),
                "--model", args.model, "--prompt-profile", prompt,
                "--langpro-builtin", "on" if wordnet else "off",
                "--max-iterations", "2", "--concurrency", str(args.concurrency), "--resume",
            ]
            if args.retry_baseline_prover_errors:
                command.append("--retry-baseline-prover-errors")
            print(f"\n[{job_num}/{total}] {prompt} | {dataset_name} | WordNet {'on' if wordnet else 'off'}")
            print("Command: " + " ".join(command))
            if args.dry_run:
                continue

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Avoid unbounded SQLite response-cache growth for long matrix runs.
            # Per-problem JSONL checkpoints below still preserve experiment data.
            env["KBPROJECTION_LANGPRO_CACHE_BACKEND"] = "none"
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\n\n=== " + " ".join(command) + " ===\n")
                log.flush()
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            if return_code != 0:
                print(f"Job stopped with exit code {return_code}. Its JSONL checkpoint is preserved.", file=sys.stderr)
                print(f"Rerun the same command to resume. Job log: {log_path}", file=sys.stderr)
                write_summary(summary, args.output_dir)
                return return_code
            summary.append(summarize(output, prompt, dataset_name, wordnet))
            write_summary(summary, args.output_dir)
            print(f"Updated summaries: {args.output_dir / 'summary.json'} and summary.csv")

    if not args.dry_run:
        print(f"\nAll jobs complete. Summary: {args.output_dir / 'summary.json'}")
        print(f"Per-problem JSONL and logs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
