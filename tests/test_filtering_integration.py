"""Configuration propagation and complete-result cache isolation, without services."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from kbprojection.models import (
    ExperimentResult, FilteringConfig, KBResult, LangProResult, NLILabel,
    NLIProblem, ProblemConfig, TestMode as PipelineTestMode,
)
from kbprojection.orchestration import (
    process_kb_examples, process_single_problem, result_cache_fingerprint,
)
from kbprojection.runners import arun_problem


def problem(hypothesis="An animal moves."):
    return NLIProblem(
        id="p1", premises=["A dog runs."], hypothesis=hypothesis,
        gold_label=NLILabel.ENTAILMENT, dataset="manual", split="test",
    )


def test_named_profiles_and_legacy_roundtrip():
    operational = FilteringConfig.operational()
    baseline = FilteringConfig.evaluation_baseline()
    assert operational.lemmatization_kind == "pos"
    assert operational.lemmatization_mode == "additive"
    assert operational.final_ph_filter
    assert baseline.lemmatization_mode == baseline.argument_alignment_mode == "off"
    assert baseline.underscores_mode == baseline.leading_preposition_mode == "replacement"
    assert not baseline.final_ph_filter
    legacy = ProblemConfig(post_process=False)
    assert legacy.filtering.underscores_mode == legacy.filtering.leading_preposition_mode == "off"
    assert legacy.filtering.lemmatization_mode == "additive"
    assert ProblemConfig.model_validate_json(legacy.model_dump_json()) == legacy
    with pytest.raises(ValueError, match="conflicts"):
        ProblemConfig(post_process=False, filtering=operational)
    with pytest.raises(ValueError):
        FilteringConfig(lemmatization_mode="replace")
    with pytest.raises(ValueError):
        FilteringConfig(argument_order_agnostic=True)
    with pytest.raises(ValueError, match="requires"):
        FilteringConfig.evaluation_baseline(use_semantic=True)


def test_pipeline_receives_resolved_filtering_and_serializes_metadata():
    config = ProblemConfig(
        filtering=FilteringConfig.evaluation_baseline(lemmatization_mode="replacement"),
        verbose=False, test_mode=PipelineTestMode.NORMALISED_KB,
    )
    with (
        patch("kbprojection.orchestration.call_llm", new_callable=AsyncMock) as llm,
        patch("kbprojection.orchestration.langpro_api_call", new_callable=AsyncMock) as prover,
        patch("kbprojection.orchestration.pipeline_filter_kb_injections") as filtering,
    ):
        llm.return_value = ["isa_wn(dog, animal)"]
        prover.return_value = LangProResult(label=NLILabel.NEUTRAL)
        filtering.return_value = [KBResult(relation="isa_wn(dog, animal)")]
        result = asyncio.run(process_single_problem(problem(), config=config))
    assert filtering.call_args.kwargs["filtering"] is config.filtering
    assert "post_process" not in filtering.call_args.kwargs
    assert result.resolved_config == config.model_dump(mode="json")
    assert result.cache_fingerprint == result_cache_fingerprint(problem(), config)
    assert ExperimentResult.model_validate_json(result.model_dump_json()).resolved_config == result.resolved_config


def test_runner_rejects_parallel_legacy_setting():
    with pytest.raises(ValueError, match="config.filtering"):
        asyncio.run(arun_problem(problem(), config=ProblemConfig(), post_process=False))


def test_fingerprint_covers_input_and_result_shaping_configuration():
    base = ProblemConfig()
    fingerprint = result_cache_fingerprint(problem(), base)
    variants = [
        ProblemConfig(model="another-model"),
        ProblemConfig(prompt_style="lasha"),
        ProblemConfig(test_mode=PipelineTestMode.NO_KB),
        ProblemConfig(filtering=FilteringConfig.evaluation_baseline()),
    ]
    assert all(result_cache_fingerprint(problem(), config) != fingerprint for config in variants)
    assert result_cache_fingerprint(problem("A dog rests."), base) != fingerprint


def test_complete_cache_hit_and_reject_tampered_metadata(tmp_path):
    item = problem()
    config = ProblemConfig(test_mode=PipelineTestMode.NO_KB, verbose=False)

    class Loader:
        def get_problem(self, _id, split):
            return item

    async def collect():
        return [result async for result in process_kb_examples(
            Loader(), config=config, split="test", problem_ids=["p1"], cache_dir=tmp_path,
        )]

    with patch("kbprojection.orchestration.langpro_api_call", new_callable=AsyncMock) as prover:
        prover.return_value = LangProResult(label=NLILabel.NEUTRAL)
        asyncio.run(collect())
        asyncio.run(collect())
        assert prover.call_count == 1
        cache_path = next(tmp_path.glob("*.json"))
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["resolved_config"]["model"] = "wrong-model"
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        asyncio.run(collect())
        assert prover.call_count == 2


def test_generation_filtering_forwarding_and_historical_resume_guard(tmp_path):
    from scripts.experiments import run_repeated_multi_reference_experiment as generation

    config = FilteringConfig.evaluation_baseline(lemmatization_mode="replacement")
    with patch.object(generation, "pipeline_filter_kb_injections", return_value=[]) as filtering:
        generation.normalize_relations(["isa_wn(dog, animal)"], problem(), filter_kb=True, filtering=config)
        assert filtering.call_args.kwargs["filtering"] is config
        filtering.reset_mock()
        raw = ["isa_wn(dog, animal)"]
        assert generation.normalize_relations(raw, problem(), filter_kb=False, filtering=config) is raw
        filtering.assert_not_called()

    historical = tmp_path / "historical.csv"
    historical.write_text("ID,prompt,model,repeat,KB,error\np1,lasha,m,1,NO_RELATION,\n", encoding="utf-8")
    before = historical.read_bytes()
    args = generation.build_parser().parse_args([
        "--output-csv", str(historical), "--prepare-only",
        "--sample-csv", str(tmp_path / "sample.csv"),
    ])
    with pytest.raises(ValueError, match="metadata"):
        asyncio.run(generation.async_main(args))
    assert historical.read_bytes() == before
    assert not (tmp_path / "sample.csv").exists()
