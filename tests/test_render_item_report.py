from pathlib import Path

import pytest

from scripts.experiments.render_item_report import render_report, validate_lex_sources


def test_report_includes_all_references_and_diagnostics(tmp_path: Path):
    item = {
        "ID": "sample-1",
        "dataset": "snli",
        "split": "dev",
        "gold_label": "entailment",
        "premise": "A dog runs.",
        "hypothesis": "An animal moves.",
        "Alternative_KB": "isa_wn(dog, animal)",
        "Ettore_KB": "isa_wn(dog, animal); isa_wn(run, move)",
        "Jorryt_KB": "",
        "Lasha_KB": "NO_RELATION",
        "Stefan_KB": "",
    }
    prediction = {
        "prompt": "lasha",
        "model": "example/model",
        "repeat": "1",
        "raw_response": "answer: entailment\nrelations: { isa_wn(dog, animal) }",
        "KB": "isa_wn(dog, animal)",
        "error": "",
    }

    report = render_report(
        item,
        prediction,
        items_csv=tmp_path / "items.csv",
        predictions_csv=tmp_path / "outputs.csv",
    )

    assert "# KB Projection item report: sample-1" in report
    assert "### Human explanation 1" in report
    assert "### Human explanation 3" in report
    assert "No lexical relation is needed." in report
    assert "| Human explanation 1 | 1 | 0 | 0 | 1.00 |" in report
    assert "Overall similarity: **1.00**." in report
    assert "This report performed no model or LangPro calls." in report


def test_report_marks_non_entailment_output_unscored(tmp_path: Path):
    item = {
        "ID": "sample-2",
        "dataset": "snli",
        "split": "dev",
        "gold_label": "entailment",
        "premise": "A dog runs.",
        "hypothesis": "An animal moves.",
        "Alternative_KB": "NO_RELATION",
    }
    prediction = {
        "prompt": "lasha",
        "model": "example/model",
        "repeat": "1",
        "raw_response": "answer: non-entailment",
        "KB": "NO_RELATION",
        "error": "",
    }

    report = render_report(
        item,
        prediction,
        items_csv=tmp_path / "items.csv",
        predictions_csv=tmp_path / "outputs.csv",
    )

    assert "This output is not scored because its final explicit answer is `non-entailment`." in report


def test_report_includes_saved_langpro_timeline(tmp_path: Path):
    item = {
        "ID": "sample-3", "dataset": "snli", "split": "dev", "gold_label": "entailment",
        "premise": "A dog runs.", "hypothesis": "An animal moves.", "Alternative_KB": "isa_wn(dog, animal)",
    }
    prediction = {"prompt": "lasha", "model": "example/model", "repeat": "1", "raw_response": "answer: entailment", "KB": "isa_wn(dog, animal)", "error": ""}
    config = {"run": "wordnet_llm_agentic", "model": "example/model", "prompt_arm": "lex", "langpro_builtin": "on"}
    record = {
        "outcome": {"pred_baseline": "neutral", "pred_final": "entailment", "solved": True, "stop_reason": "solved"},
        "timeline": [
            {"phase": "baseline_langpro", "output": {"pred": "neutral", "prover_error": None}},
            {"phase": "kb_generation", "attempt": 1, "output": {"kb_raw": ["isa_wn(dog, animal)"]}},
            {"phase": "kb_filtering", "attempt": 1, "output": {"kb_filtered": ["isa_wn(dog, animal)"]}},
            {"phase": "langpro_with_kb", "attempt": 1, "output": {"pred": "entailment", "kb_verification": {"all_sent_found_in_response": True}, "proof_excerpts": {"entailment": {"closed": True}, "contradiction": {"closed": False}}}},
        ],
    }

    report = render_report(
        item, prediction, items_csv=tmp_path / "items.csv", predictions_csv=tmp_path / "outputs.csv",
        langpro_config=config, langpro_record=record, langpro_results_json=tmp_path / "langpro.json",
    )

    assert "## Saved LangPro proof run" in report
    assert "| Prompt | Lasha Plus Precision (agentic KB format) |" in report
    assert "### Attempt 1" in report
    assert "- Entailment branch closed: True" in report


def test_report_rejects_other_prompt_families():
    with pytest.raises(ValueError, match="Lasha Plus Precision"):
        validate_lex_sources({"prompt": "ettore"}, None)
    with pytest.raises(ValueError, match="Lex agentic form"):
        validate_lex_sources({"prompt": "lasha"}, {"prompt_arm": "stefan"})
