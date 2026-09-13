"""Read-only reconciliation of the expanded LangPro tables (Python stdlib only).

Run beside the archived exports: python3 reconcile_proof_coverage.py
One row per population/model/prompt; no averaging across exports or populations.
"""

import json
from collections import Counter
from pathlib import Path

CONFIGURATIONS = (
    "wordnet_only", "llm_only_one_shot", "llm_only_agentic",
    "wordnet_llm_one_shot", "wordnet_llm_agentic",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def problem_id(record):
    return record.get("problem_id", record.get("id"))


def gold(record):
    return record.get("problem", record)["gold_label"]


def solved(record):
    outcome = record["outcome"]
    return outcome == "solved" if isinstance(outcome, str) else outcome["solved"] is True


def validate(records, population, labels):
    ids = [problem_id(r) for r in records]
    assert len(ids) == population and None not in ids
    assert len(set(ids)) == population, "Duplicate problem IDs"
    assert dict(Counter(gold(r) for r in records)) == labels
    for r in records:
        pred = r["pred"] if isinstance(r["outcome"], str) else r["outcome"]["pred_final"]
        assert solved(r) == (pred == gold(r)), (problem_id(r), pred, gold(r))
    return set(ids)


def first_attempt_solved(record):
    """Use baseline success or attempt 1, never a filename or attempt 2."""
    events = [t for t in record["timeline"] if t["phase"] == "baseline_langpro"
              or (t["phase"] == "langpro_with_kb" and t.get("attempt") == 1)]
    for event in events:
        output = event.get("output", {})
        if output.get("matches_gold") is not None:
            assert output["matches_gold"] == (output.get("pred") == gold(record))
    return any(t.get("output", {}).get("matches_gold") is True for t in events)


def table_row(model, prompt, population, counts, sources, version):
    return dict(model=model, prompt=prompt, population=population, version=version,
                wordnet=counts[0], llm_one_shot=counts[1], llm_agentic=counts[2],
                llm_delta=counts[2]-counts[1], wordnet_llm_one_shot=counts[3],
                wordnet_llm_agentic=counts[4], wordnet_llm_delta=counts[4]-counts[3],
                sources=sources)


def audit(root):
    root = Path(root)
    metrics = {(m["model"], m["prompt"], m["configuration"], m["total"]): m
               for m in read(root / "final_metrics.json")}
    populations = {}
    rows, errors, baselines_only = [], [], []
    for n in (1000, 365):
        labels = {"entailment": 1000} if n == 1000 else {
            "entailment": 363, "contradiction": 1, "neutral": 1}
        for arm in sorted((root / f"current-{n}").glob("*/*")):
            files = [arm / (name + ".json") for name in CONFIGURATIONS]
            if not all(f.exists() for f in files):
                assert [f.name for f in files if f.exists()] == ["wordnet_only.json"]
                baselines_only.append(str(arm.relative_to(root)))
                continue
            counts, source_paths, config_ids = [], [], []
            for name, path in zip(CONFIGURATIONS, files):
                data = read(path)
                ids = validate(data["records"], n, labels)
                populations.setdefault(n, ids)
                assert ids == populations[n], "Population differs across configurations"
                config_ids.append(ids)
                count = sum(solved(r) for r in data["records"])
                model = data["config"]["model"]
                metric = metrics[(model, arm.name, name, n)]
                assert count == metric["correct"]
                assert abs(metric["accuracy"] - 100 * count/n) < 0.00001
                counts.append(count)
                source_paths.append(str(path.relative_to(root)))
                errors.append(dict(model=model, prompt=arm.name, population=n,
                                   configuration=name, llm_errors=metric["llm_errors"],
                                   langpro_errors=metric["langpro_errors"], unknown=metric["unknown"]))
            rows.append(table_row(model, arm.name, n, counts, source_paths, "recent-protocol"))

    gemini_dir = root / "gemini-3.1-flash-lite-jorryt-new-prompt-1000"
    gemini = {}
    for name in ("wordnet-only", "llm-only-first-run", "llm-only-second-run", "llm-and-wordnet"):
        path = gemini_dir / (name + ".json")
        data = read(path)
        assert validate(data["records"], 1000, {"entailment": 1000}) == populations[1000]
        if name != "wordnet-only":
            assert data["config"]["max_iterations"] == 2
        gemini[name] = data["records"]
    baseline = sum(solved(r) for r in gemini["wordnet-only"])
    export_versions = []
    for name in ("llm-only-first-run", "llm-only-second-run"):
        export_versions.append(dict(file=name + ".json", one_shot=sum(map(first_attempt_solved, gemini[name])),
                                    agentic=sum(map(solved, gemini[name]))))
    counts = [baseline, sum(map(first_attempt_solved, gemini["llm-only-second-run"])),
              sum(map(solved, gemini["llm-only-second-run"])),
              sum(map(first_attempt_solved, gemini["llm-and-wordnet"])),
              sum(map(solved, gemini["llm-and-wordnet"]))]
    rows.insert(0, table_row("google/gemini-3.1-flash-lite", "lex", 1000, counts,
                            [str((gemini_dir/(name+".json")).relative_to(root)) for name in
                             ("wordnet-only", "llm-only-second-run", "llm-and-wordnet")], "jorryt-v2"))
    old = {problem_id(r): r for r in gemini["llm-only-first-run"]}
    new = {problem_id(r): r for r in gemini["llm-only-second-run"]}
    export_disagreements = [dict(problem_id=i, first_export_one_shot=first_attempt_solved(old[i]),
                                second_export_one_shot=first_attempt_solved(new[i]),
                                first_export_agentic=solved(old[i]), second_export_agentic=solved(new[i]))
                            for i in old if first_attempt_solved(old[i]) != first_attempt_solved(new[i])
                            or solved(old[i]) != solved(new[i])]
    recent_path = root / "current-1000/openai__gpt-oss-20b/lex/wordnet_only.json"
    recent_baseline = {r["id"]: r for r in read(recent_path)["records"]}
    baseline_disagreements = [dict(problem_id=r["id"], jorryt_pred=r["pred"],
                                   recent_pred=recent_baseline[r["id"]]["pred"],
                                   recent_category=recent_baseline[r["id"]].get("category"))
                              for r in gemini["wordnet-only"]
                              if solved(r) != solved(recent_baseline[r["id"]])]
    return dict(rows=rows, gemini_export_versions=export_versions,
                gemini_export_disagreements=export_disagreements,
                baseline_disagreements=baseline_disagreements,
                baseline_only_folders_not_treated_as_experiments=baselines_only,
                technical_outcomes=errors,
                checks={"unique_ids_and_complete_populations": True,
                        "same_ids_across_1000_exports_including_gemini": True,
                        "solved_equals_gold_label_agreement": True,
                        "current_counts_match_final_metrics": True,
                        "labels_1000": {"entailment": 1000},
                        "labels_365": {"entailment": 363, "contradiction": 1, "neutral": 1}})


if __name__ == "__main__":
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2))
