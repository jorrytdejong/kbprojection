"""CLI commands for the agentic LangPro pipeline."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from kbprojection.models import NLILabel, NLIProblem

from agentic_pipeline_langpro.llm_config import resolve_llm
from agentic_pipeline_langpro.paths import DATA_DIR, RESULTS_DIR, load_repo_dotenv
from agentic_pipeline_langpro.pipeline import arun_agentic_problem
from agentic_pipeline_langpro.prompts import prompt_sha256
from agentic_pipeline_langpro.results import build_run_record, format_run_report

DEFAULT_INPUT = DATA_DIR / "snli_train_entailment_1k.jsonl"
NETWORK_RETRY_ATTEMPTS = 5
NETWORK_RETRY_BASE_SECONDS = 2
TRANSIENT_NETWORK_MARKERS = (
    "nodename nor servname",
    "name or service not known",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "network is unreachable",
    "connection error",
    "connection refused",
    "connection reset",
    "server disconnected",
    "timed out",
    "timeout",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)


class NetworkInterruptionError(RuntimeError):
    """Raised when transient network failures persist after bounded retries."""


def _network_error_from_meta(meta: Any) -> Optional[str]:
    errors = [getattr(meta, "baseline_error", None)]
    errors.extend(
        error
        for iteration in getattr(meta, "iterations", [])
        for error in (getattr(iteration, "llm_error", None), getattr(iteration, "langpro_error", None))
    )
    for error in errors:
        if not error:
            continue
        message = str(error)
        lowered = message.lower()
        if any(marker in lowered for marker in TRANSIENT_NETWORK_MARKERS):
            return message
    return None


def _load_checkpoint(path: Path, expected_config: dict, expected_ids: set[str]) -> dict[str, dict]:
    """Read completed rows; preserve a damaged, unterminated tail before recovery."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if not raw:
        return {}
    lines = raw.splitlines(keepends=True)
    records: dict[str, dict] = {}
    valid_bytes = 0
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if is_last and not line.endswith((b"\n", b"\r")):
                backup = path.with_name(path.name + ".partial_tail.bak")
                if backup.exists():
                    raise ValueError(f"Cannot recover partial final line: backup already exists at {backup}")
                backup.write_bytes(line)
                with path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
                print(f"Recovered incomplete final JSONL line to {backup}", file=sys.stderr)
                break
            raise ValueError(f"Invalid JSONL at line {index + 1} in {path}")
        if not isinstance(record, dict) or not (record.get("problem_id") or record.get("id")):
            raise ValueError(f"Checkpoint line {index + 1} has no problem id")
        problem_id = str(record.get("problem_id") or record["id"])
        if problem_id not in expected_ids:
            raise ValueError(f"Checkpoint contains id not present in this input: {problem_id}")
        if problem_id in records:
            raise ValueError(f"Duplicate id in checkpoint: {problem_id}")
        if record.get("run_config") != expected_config:
            raise ValueError(
                f"Checkpoint configuration differs at id {problem_id}; use a new --output path."
            )
        records[problem_id] = record
        valid_bytes += len(line)
    return records


def load_problems(path: Path, limit: Optional[int] = None) -> List[NLIProblem]:
    rows: List[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    problems: List[NLIProblem] = []
    for row in rows:
        extra = {
            key: row[key]
            for key in ("pred_baseline", "baseline_error")
            if key in row and row[key] is not None
        }
        problems.append(
            NLIProblem(
                id=str(row["id"]),
                premises=[str(row["premise"])],
                hypothesis=str(row["hypothesis"]),
                gold_label=NLILabel(str(row["gold_label"]).lower()),
                dataset=str(row.get("dataset", "snli")),
                split=str(row.get("split", "train")),
                original_data=extra or None,
            )
        )
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic LangPro pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-profile", choices=("lex", "stefan"), default="lex")
    parser.add_argument("--resume", action="store_true", help="Resume a matching JSONL checkpoint.")
    parser.add_argument(
        "--retry-baseline-prover-errors",
        action="store_true",
        help="On resume, retry rows whose baseline LangPro call failed; save the original checkpoint beside it.",
    )
    parser.add_argument("--langpro-builtin", choices=("on", "off"), default="on")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also write a pretty-printed .json file next to the JSONL output.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Also write a markdown .md summary next to the JSONL output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_repo_dotenv()
    os.environ["KBPROJECTION_LANGPRO_PROVER_CONFIG_EXTRA"] = (
        "" if args.langpro_builtin == "on" else "no_kb,no_wn"
    )

    input_path = args.input.resolve()
    problems = load_problems(input_path, args.limit)
    if not problems:
        print("No problems loaded.", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"agentic_{ts}.jsonl"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = resolve_llm(model=args.model)
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    run_config = {
        "input_sha256": input_sha,
        "prompt_profile": args.prompt_profile,
        "prompt_sha256": prompt_sha256(args.prompt_profile),
        "model": llm.model,
        "wordnet": args.langpro_builtin == "on",
        "max_iterations": args.max_iterations,
    }
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    print(f"Loaded {len(problems)} problems from {input_path}")
    print(f"LLM (OpenRouter): {llm.model}")
    print(f"LangPro WordNet: {args.langpro_builtin}")
    print(f"Prompt profile: {args.prompt_profile} ({run_config['prompt_sha256'][:12]})")
    print(f"Output: {out_path}")

    expected_ids = {problem.id for problem in problems}
    if len(expected_ids) != len(problems):
        raise ValueError("Input contains duplicate problem ids; refusing unsafe checkpointing.")
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {out_path}; choose another path or pass --resume")
    completed = _load_checkpoint(out_path, run_config, expected_ids) if args.resume else {}
    if args.retry_baseline_prover_errors and completed:
        retry_ids = {
            problem_id
            for problem_id, record in completed.items()
            if record.get("outcome", {}).get("baseline_outcome") == "prover_error"
            or record.get("outcome", {}).get("final_status") == "baseline_prover_failed"
        }
        if retry_ids:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = out_path.with_name(f"{out_path.name}.before_retry_{timestamp}")
            import shutil

            shutil.copy2(out_path, backup_path)
            completed = {
                problem_id: record
                for problem_id, record in completed.items()
                if problem_id not in retry_ids
            }
            temp_path = out_path.with_name(f"{out_path.name}.retry_tmp_{timestamp}")
            with temp_path.open("w", encoding="utf-8") as handle:
                for record in completed.values():
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, out_path)
            print(
                f"Removed {len(retry_ids)} baseline-prover-error rows for retry; "
                f"original checkpoint backed up to {backup_path}",
                file=sys.stderr,
            )
    if completed and out_path.exists() and not out_path.read_bytes().endswith(b"\n"):
        with out_path.open("ab") as handle:
            handle.write(b"\n")
    pending = [(i, p) for i, p in enumerate(problems) if p.id not in completed]
    print(f"Checkpoint: {len(completed)} complete, {len(pending)} remaining")
    checkpoint_handle = out_path.open("a", encoding="utf-8")

    async def run_all() -> List[dict]:
        semaphore = asyncio.Semaphore(args.concurrency)
        checkpoint_lock = asyncio.Lock()
        records: List[Optional[dict]] = [completed.get(p.id) for p in problems]
        done = len(completed)
        start = time.monotonic()

        async def one(slot: int, problem: NLIProblem) -> None:
            nonlocal done
            async with semaphore:
                for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
                    result, meta = await arun_agentic_problem(
                        problem,
                        model=llm.model,
                        max_iterations=args.max_iterations,
                        prompt_profile=args.prompt_profile,
                    )
                    network_error = _network_error_from_meta(meta)
                    if network_error is None:
                        break
                    if attempt == NETWORK_RETRY_ATTEMPTS:
                        raise NetworkInterruptionError(
                            f"Network error persisted for {problem.id} after "
                            f"{NETWORK_RETRY_ATTEMPTS} attempts: {network_error}"
                        )
                    delay = NETWORK_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"\n[network] transient connection failure for {problem.id}; "
                        f"retry {attempt + 1}/{NETWORK_RETRY_ATTEMPTS} in {delay}s: "
                        f"{network_error}",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(delay)
            record = build_run_record(problem, result, meta)
            record["run_config"] = run_config
            records[slot] = record
            async with checkpoint_lock:
                checkpoint_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                checkpoint_handle.flush()
                os.fsync(checkpoint_handle.fileno())
            done += 1
            elapsed = int(time.monotonic() - start)
            sys.stdout.write(f"\r{done}/{len(problems)} | {elapsed}s")
            sys.stdout.flush()
            if done == len(problems):
                sys.stdout.write("\n")

        await asyncio.gather(*(one(i, p) for i, p in pending))
        return [r for r in records if r is not None]

    try:
        records = asyncio.run(run_all())
    except NetworkInterruptionError as exc:
        print(
            f"\n{exc}\nStopping without checkpointing this problem. "
            "Completed JSONL rows are preserved; restore the connection and rerun with --resume.",
            file=sys.stderr,
        )
        return 75
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        checkpoint_handle.close()

    records = [r for r in records if r is not None]
    print(f"Checkpoint contains {len(records)} records at {out_path}")

    if args.pretty:
        pretty_path = out_path.with_suffix(".json") if out_path.suffix.lower() != ".json" else out_path
        pretty_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote pretty JSON to {pretty_path}")

    if args.write_report:
        report_path = out_path.with_suffix(".md")
        report_path.write_text(
            "\n\n---\n\n".join(format_run_report(r) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
