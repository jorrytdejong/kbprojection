"""Shared neutral helpers for offline KB scoring replays."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

from calculate_multi_reference_f1 import item_selection_score, relation_counts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COLUMNS = ["Alternative_KB", "Ettore_KB", "Jorryt_KB", "Lasha_KB", "Stefan_KB"]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input has no CSV header: {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def source_sha256(path: Path) -> str:
    """Hash canonical LF source, matching Git's text blobs on every platform."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_source_provenance(code_hashes: dict[str, str]) -> dict:
    """Anchor source fingerprints to a commit when Git metadata is available."""
    command = ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", "-C", str(REPOSITORY_ROOT)]
    try:
        commit = subprocess.check_output(command + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        differences = []
        for name, expected in code_hashes.items():
            try:
                content = subprocess.check_output(command + ["show", f"{commit}:{name}"], stderr=subprocess.DEVNULL)
                actual = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
            except subprocess.CalledProcessError:
                actual = None
            if actual != expected:
                differences.append(name)
        return {"commit": commit, "code_matches_commit": not differences,
                "different_or_untracked_code_paths": differences}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "code_matches_commit": None,
                "reason": "Local Git metadata is unavailable; use code_sha256 to identify the source."}


def choose_reference(prediction: frozenset, references: list[tuple[str, frozenset]]):
    """Use the existing scorer's counts, item F1, and exact reference tie-break."""
    scored = [(item_selection_score(counts), column, reference, counts)
              for column, reference in references
              for counts in [relation_counts(prediction, reference)]]
    return max(scored, key=lambda entry: (
        entry[0], entry[3].tp, -entry[3].fp, -entry[3].fn, entry[1],
    ))


def prepare_slots(output_rows: list[dict], source_rows: list[dict], repeats: int):
    """Validate joins and synthesize absent slots; never turn missing into NO_RELATION."""
    source_by_id = {}
    for source in source_rows:
        identifier = source.get("ID", "")
        if not identifier or identifier in source_by_id:
            raise ValueError(f"Missing or duplicate source ID: {identifier!r}")
        source_by_id[identifier] = source
    if not source_by_id:
        raise ValueError("The reference CSV contains no items.")
    indexed, groups = {}, set()
    for row in output_rows:
        identifier, prompt, model = row.get("ID", ""), row.get("prompt", ""), row.get("model", "")
        repeat = int(row.get("repeat", ""))
        if not prompt or not model or not 1 <= repeat <= repeats:
            raise ValueError(f"Invalid prompt/model/repeat in output row: {(identifier, prompt, model, repeat)}")
        key = (prompt, model, repeat, identifier)
        if key in indexed:
            raise ValueError(f"Duplicate prediction slot: {key}")
        if identifier not in source_by_id:
            raise ValueError(f"Prediction ID has no source item: {identifier}")
        source = source_by_id[identifier]
        for field in ("premise", "hypothesis"):
            if row.get(field, "") != source.get(field, ""):
                raise ValueError(f"{field} mismatch for prediction slot {key}")
        indexed[key] = row
        groups.add((prompt, model))
    if not groups:
        raise ValueError("The saved-output CSV contains no model/prompt groups.")
    return [(prompt, model, repeat, source, indexed.get((prompt, model, repeat, identifier)))
            for prompt, model in sorted(groups) for repeat in range(1, repeats + 1)
            for identifier, source in source_by_id.items()]


def summarize(metrics: list[dict]) -> list[dict]:
    groups = {}
    for row in metrics:
        groups.setdefault((row["prompt"], row["model"], row["configuration"]), []).append(row)
    result = []
    for (prompt, model, name), rows in sorted(groups.items()):
        row = {"prompt": prompt, "model": model, "configuration": name, "generation_runs": len(rows)}
        for metric in ("precision", "recall", "micro_f1", "exact_best_match_rate"):
            values = [r[metric] for r in rows if math.isfinite(r[metric])]
            row[f"runs_with_{metric}"] = len(values)
            row[f"mean_{metric}"] = statistics.mean(values) if values else math.nan
            row[f"sample_stddev_{metric}"] = statistics.stdev(values) if len(values) > 1 else math.nan
        for metric in ("total_items", "evaluated_items", "error_runs", "skipped_missing_prediction", "skipped_no_reference"):
            row[f"total_{metric}"] = sum(r[metric] for r in rows)
        result.append(row)
    return result


def resource_fingerprints(resources: dict) -> dict:
    result = {}
    for name, location in resources.items():
        path = Path(str(location))
        entry = {"location": str(location)}
        if path.is_file():
            entry["sha256"] = sha256(path)
        elif path.is_dir():
            hashes = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
            entry["files"] = hashes
            entry["sha256"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        else:
            match = re.match(r"^(.*?\.zip)(?:[/\\].*)?$", str(location))
            if not match or not Path(match[1]).is_file():
                raise ValueError(f"Cannot fingerprint NLP resource {name}: {location}")
            entry["archive_sha256"] = sha256(Path(match[1]))
        result[name] = entry
    return result
