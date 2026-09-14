"""Reproduce the Flash-Lite contrasts from archived exports; no API calls."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "google/gemini-3.1-flash-lite"
CONTRASTS = (
    ("wordnet_only", "llm_only_one_shot"),
    ("wordnet_only", "llm_only_agentic"),
    ("wordnet_llm_one_shot", "wordnet_llm_agentic"),
)
EXPECTED = {"lex": (26, 39, 14), "stefan": (29, 38, 16)}


def pid(record):
    return record.get("problem_id", record.get("id"))


def solved(record):
    value = record["outcome"]
    return value == "solved" if isinstance(value, str) else value["solved"]


def system(record, configuration):
    wn = configuration.startswith("wordnet_")
    common = {
        "configuration": configuration,
        "wordnet_enabled": wn,
        "wordnet_internal_relations": None,
        "wordnet_internal_relations_note": (
            "Built-in WordNet is enabled; its internal relations are not enumerated in this export."
            if wn else "Built-in WordNet is disabled."
        ),
        "solved": solved(record),
    }
    if isinstance(record["outcome"], str):
        return dict(common, prediction=record["pred"], category=record["category"],
                    kb=[], kb_scope="Additional injected relations only; not the built-in WordNet KB.",
                    selected_attempt=0, kb_source="WordNet baseline; no additional KB injected.",
                    attempts=[], source_problem_id=pid(record))
    outcome = record["outcome"]
    attempts = []
    for event in record["timeline"]:
        if event["phase"] != "langpro_with_kb":
            continue
        attempts.append({
            "attempt": event["attempt"],
            "kb": event["input"]["kb_sent"],
            "prediction": event["output"]["pred"],
            "proof_info": event["output"].get("proof_excerpts"),
        })
    last_attempt = outcome["attempt_count"]
    selected = next((a for a in attempts if a["attempt"] == last_attempt), None)
    if selected is not None:
        assert selected["prediction"] == outcome["pred_final"], pid(record)
        kb = selected["kb"]
        source = "timeline: langpro_with_kb.input.kb_sent at selected_attempt"
    else:
        # A skipped/empty KB uses the baseline, not an invented prover attempt.
        filtered = [e for e in record["timeline"]
                    if e["phase"] == "kb_filtering" and e.get("attempt") == last_attempt]
        assert last_attempt == 0 or (filtered and filtered[-1]["output"]["kb_filtered"] == []), pid(record)
        assert outcome["pred_final"] == outcome["pred_baseline"], pid(record)
        kb = []
        source = "Baseline retained; no additional KB evaluated at the final attempt."
    return dict(common, prediction=outcome["pred_final"], category=outcome["category"],
                kb=kb, kb_scope="Additional injected relations only; not the built-in WordNet KB.",
                selected_attempt=last_attempt, kb_source=source, attempts=attempts,
                stop_reason=outcome.get("stop_reason"), provenance=record.get("provenance"))


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    index = []
    for prompt in ("lex", "stefan"):
        folder = ROOT / "results/1000/google__gemini-3.1-flash-lite" / prompt
        data, hashes = {}, {}
        for name in {x for pair in CONTRASTS for x in pair}:
            path = folder / (name + ".json")
            raw = path.read_bytes()
            export = json.loads(raw)
            assert export["config"]["model"] == MODEL
            records = {pid(r): r for r in export["records"]}
            assert len(records) == len(export["records"]) == 1000
            data[name] = records
            hashes[name] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        for position, (before, after) in enumerate(CONTRASTS):
            assert data[before].keys() == data[after].keys()
            records = []
            for problem_id, failed in data[before].items():
                succeeded = data[after][problem_id]
                if solved(failed) or not solved(succeeded):
                    continue
                problem = succeeded["problem"]
                old_problem = failed.get("problem", failed)
                premise = problem["premise"]
                old_premise = old_problem["premise"]
                assert premise == (old_premise if isinstance(old_premise, list) else [old_premise])
                assert problem["hypothesis"] == old_problem["hypothesis"]
                assert problem["gold_label"] == old_problem["gold_label"] == "entailment"
                records.append({
                    "problem_id": problem_id,
                    "premise": premise[0] if len(premise) == 1 else premise,
                    "hypothesis": problem["hypothesis"],
                    "gold_label": problem["gold_label"],
                    "unsolved_system": system(failed, before),
                    "solved_system": system(succeeded, after),
                })
            assert len(records) == EXPECTED[prompt][position]
            filename = f"flash-lite_1000_{prompt}__{before}_unsolved__{after}_solved.json"
            metadata = {
                "model": MODEL, "population": 1000, "prompt": prompt,
                "prompt_number": 1 if prompt == "lex" else 2,
                "run": "Ettore shared-protocol run; excludes the historical Jorryt reference run",
                "unsolved_configuration": before, "solved_configuration": after,
                "selection": "Same problem ID; first system solved=false and second system solved=true.",
                "count": len(records),
                "sources": [{"file": (folder / (name + ".json")).relative_to(ROOT).as_posix(),
                             "sha256_lf": hashes[name]} for name in (before, after)],
                "notes": [
                    "kb contains additional injected relations. An empty list does not mean built-in WordNet is empty.",
                    "One-shot is baseline plus attempt 1 of the shared agentic execution, not an independent run.",
                    "Unsolved includes unknown outcomes; category and prediction are preserved.",
                    "Cases may overlap across contrasts. No judgment of lexical validity is inferred from proof success.",
                ],
            }
            write_json(HERE / filename, {"metadata": metadata, "records": records})
            index.append({"file": filename, "prompt": prompt, "unsolved_system": before,
                          "solved_system": after, "count": len(records)})
    write_json(HERE / "index.json", {"model": MODEL, "population": 1000, "files": index})
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
