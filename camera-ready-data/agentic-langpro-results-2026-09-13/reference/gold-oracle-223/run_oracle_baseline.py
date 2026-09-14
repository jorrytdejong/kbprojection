#!/usr/bin/env python3
"""Evaluate LangPro with agreed human lexical-entailment annotations.

This is an oracle baseline: it bypasses the LLM and injects the annotators'
relations directly into LangPro.  The primary result deliberately does *not*
use kbprojection's normalisation/filtering pipeline, because that pipeline can
create lemma and direction-swapped variants of an annotation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

HUMAN_COLUMNS = ("Ettore_KB", "Jorryt_KB", "Lasha_KB", "Stefan_KB")
NO_RELATION_VALUES = {
    "",
    "()",
    "(no, relation)",
    "(no relation)",
    "no relation",
    "no_relation",
    "none",
    "(none)",
}


def parse_annotation(value: str) -> tuple[tuple[str, str], ...]:
    """Parse ``(premise term, hypothesis term); ...`` into canonical pairs."""
    text = str(value or "").strip().lower().replace("?", "")
    if text in NO_RELATION_VALUES:
        return ()

    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"\(([^()]*)\)", text):
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Expected a two-place relation, found {match.group(0)!r}")
        pairs.append((parts[0], parts[1]))
    if not pairs:
        raise ValueError(f"Could not parse an annotation: {value!r}")
    return tuple(sorted(set(pairs)))


def agreed_relations(row: dict[str, str], *, min_annotators: int) -> tuple[tuple[str, str], ...] | None:
    """Return the shared annotation, or None when the row is not an oracle item."""
    completed = [
        parse_annotation(row.get(column, ""))
        for column in HUMAN_COLUMNS
        if str(row.get(column, "")).strip()
    ]
    if len(completed) < min_annotators:
        return None
    if len(set(completed)) != 1:
        return None
    return completed[0]


def as_langpro_kb(relations: Iterable[tuple[str, str]]) -> list[str]:
    return [f"isa_wn({left}, {right})" for left, right in relations]


async def evaluate_row(row: dict[str, str], kb: list[str], semaphore: asyncio.Semaphore) -> dict:
    from kbprojection.langpro import langpro_api_call

    async with semaphore:
        baseline, oracle = await asyncio.gather(
            langpro_api_call([row["premise"]], row["hypothesis"], report=False),
            langpro_api_call([row["premise"]], row["hypothesis"], kb=kb, report=False),
        )

    gold = row["gold_label"].strip().lower()
    baseline_pred = baseline.label.value if baseline.label is not None else None
    oracle_pred = oracle.label.value if oracle.label is not None else None
    return {
        "id": row["ID"].strip(),
        "dataset": row.get("dataset", "").strip(),
        "split": row.get("split", "").strip(),
        "premise": row["premise"],
        "hypothesis": row["hypothesis"],
        "gold_label": gold,
        "human_relations": [list(pair) for pair in parse_annotation_to_pairs(kb)],
        "kb_sent": kb,
        "baseline": {"prediction": baseline_pred, "correct": baseline_pred == gold, "error": baseline.error},
        "oracle": {"prediction": oracle_pred, "correct": oracle_pred == gold, "error": oracle.error},
    }


def parse_annotation_to_pairs(kb: Iterable[str]) -> Iterable[tuple[str, str]]:
    """Recover readable pairs from the exact relation strings sent to LangPro."""
    for relation in kb:
        match = re.fullmatch(r"isa_wn\((.*), (.*)\)", relation)
        if match is None:
            raise ValueError(f"Unexpected LangPro relation {relation!r}")
        yield match.group(1), match.group(2)


def select_rows(path: Path, min_annotators: int) -> tuple[list[dict[str, str]], Counter[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ID", "premise", "hypothesis", "gold_label", *HUMAN_COLUMNS}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")

    skipped: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    for row in rows:
        if not row.get("gold_label", "").strip():
            skipped["missing_gold_label"] += 1
            continue
        if agreed_relations(row, min_annotators=min_annotators) is None:
            skipped["no_exact_human_agreement"] += 1
            continue
        selected.append(row)
    return selected, skipped


async def main_async() -> int:
    default_input = REPO_ROOT.parent / "annotator agreement - IAA_overview_edit.csv"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Run LangPro with agreed human lexical relations.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / f"gold_lex_oracle_{timestamp}.json")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-annotators", type=int, default=2)
    args = parser.parse_args()
    if args.min_annotators < 1:
        parser.error("--min-annotators must be at least 1")

    rows, skipped = select_rows(args.input.resolve(), args.min_annotators)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No eligible oracle rows found.")

    semaphore = asyncio.Semaphore(args.concurrency)
    records = await asyncio.gather(
        *(evaluate_row(row, as_langpro_kb(agreed_relations(row, min_annotators=args.min_annotators) or ()), semaphore) for row in rows)
    )
    summary = {
        "problems": len(records),
        "baseline_solved": sum(record["baseline"]["correct"] for record in records),
        "oracle_solved": sum(record["oracle"]["correct"] for record in records),
        "oracle_added_over_baseline": sum(
            not record["baseline"]["correct"] and record["oracle"]["correct"] for record in records
        ),
        "oracle_failed": sum(not record["oracle"]["correct"] for record in records),
    }
    payload = {
        "config": {
            "purpose": "gold lexical-entailment oracle baseline",
            "input": str(args.input.resolve()),
            "min_annotators": args.min_annotators,
            "filtering": "disabled; exact human relations only",
            "relation_format": "(x, y) converted only to isa_wn(x, y)",
            "skipped_rows": dict(skipped),
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
