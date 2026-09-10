import asyncio
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, List, Set, Optional, Iterator
from .async_runtime import AsyncRunContext
from .models import ExperimentResult, ExperimentStepStatus, ExperimentStatus, NLIProblem, NLILabel, ProblemConfig, TestMode, LangProResult
from .loaders.base import DatasetLoader
from .langpro import langpro_api_call
from .llm import call_llm
from .filtering import filter_kb_by_prem_hyp

from .filtering import FILTERING_PIPELINE_VERSION, pipeline_filter_kb_injections


def result_cache_fingerprint(prob: NLIProblem, config: ProblemConfig) -> str:
    """Key complete results by input content, resolved settings and pipeline version."""
    payload = {
        "schema": 2,
        "filtering_version": FILTERING_PIPELINE_VERSION,
        "problem": prob.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(_to_jsonable(key)): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def experiment_result_to_json(result: ExperimentResult, *, indent: Optional[int] = None) -> str:
    return json.dumps(_to_jsonable(result), indent=indent, ensure_ascii=False)


def _save_cache_result(cache_path: Path, result: ExperimentResult):
    """Helper to save a result object to a JSON file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = experiment_result_to_json(result, indent=2)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"[cache] Result cached to {cache_path}")
    except Exception as e:
        print(f"[Warning] Failed to save cache to {cache_path}: {e}")


async def process_single_problem(
    prob: NLIProblem,
    config: Optional[ProblemConfig] = None,
    cache_file: Optional[Path] = None,
    context: Optional[AsyncRunContext] = None,
    baseline_no_kb: Optional[LangProResult] = None,
) -> ExperimentResult:
    """
    Process a single NLI problem through the KB injection pipeline.
    
    Args:
        prob: The NLI problem to process.
        config: Configuration for processing. If None, uses defaults.
        cache_file: Optional path to save/load cached results.
    
    Returns:
        ExperimentResult with all pipeline steps populated.
    """
    if config is None:
        config = ProblemConfig()
    
    def log(msg: str):
        if config.verbose:
            print(msg)
    
    test_mode = config.test_mode.value if isinstance(config.test_mode, TestMode) else config.test_mode
    
    log(f"\n[process] Key: {prob.id} | Gold: {prob.gold_label.value}")
    log(f"[process] Premises: {prob.premises}")
    log(f"[process] Hypothesis: {prob.hypothesis}")
    log(f"[process] Mode: {test_mode} | Ablation: {config.run_ablation}")

    exp_result = ExperimentResult(
        problem=prob, prover_calls=[],
        resolved_config=config.model_dump(mode="json"),
        filtering_version=FILTERING_PIPELINE_VERSION,
        cache_fingerprint=result_cache_fingerprint(prob, config),
    )

    # ----- Step 1: No-KB baseline -----
    if baseline_no_kb is None:
        log("  [no-KB] Calling LangPro...")
        lp_res_no_kb = await langpro_api_call(prob.premises, prob.hypothesis, report=False, context=context)
    else:
        log("  [no-KB] Reusing provided LangPro baseline.")
        lp_res_no_kb = baseline_no_kb

    exp_result.pred_no_kb = lp_res_no_kb.label
    exp_result.prover_calls.append(lp_res_no_kb)
    log(f"  [no-KB] Predicted: {lp_res_no_kb.label.value}")

    if lp_res_no_kb.error:
        log("  [no-KB] Call failed.")
        exp_result.status_no_kb = ExperimentStepStatus.ERROR
        exp_result.final_status = ExperimentStatus.BASELINE_PROVER_FAILED
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result
         
    exp_result.status_no_kb = ExperimentStepStatus.SUCCESS

    # Early exit if only testing no-KB
    if test_mode == "no_kb":
        if lp_res_no_kb.label == prob.gold_label:
            exp_result.final_status = ExperimentStatus.BASELINE_SOLVED
        else:
            exp_result.final_status = ExperimentStatus.UNKNOWN # Or a specific WRONG status
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    if lp_res_no_kb.label == prob.gold_label:
        log("  [no-KB] Already correct.")
        exp_result.final_status = ExperimentStatus.BASELINE_SOLVED
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    if lp_res_no_kb.label != NLILabel.NEUTRAL:
        log("  [no-KB] Wrong and not neutral.")
        exp_result.final_status = ExperimentStatus.KB_NOT_SOLVED
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 2: Generate KB -----
    try:
        kb_raw = await call_llm(config.llm_provider, config.model, config.prompt_style, prob, context=context)
    except Exception as e:
        log(f"  [KB] LLM Call failed: {e}")
        exp_result.llm_error = str(e)
        exp_result.final_status = ExperimentStatus.KB_GENERATION_FAILED
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result
        
    exp_result.kb_raw = kb_raw
    log(f"  [KB raw] Proposed: {kb_raw}")

    if not kb_raw:
        log("  [KB] Empty KB.")
        exp_result.final_status = ExperimentStatus.KB_GENERATION_EMPTY
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 3: Test with RAW KB (if requested) -----
    raw_kb_fixed = False
    lp_res_raw: Optional[LangProResult] = None
    raw_task = None
    if test_mode in ("raw_kb", "both"):
        log("  [raw-KB] Calling LangPro with raw (unfiltered) KB...")
        raw_task = asyncio.create_task(
            langpro_api_call(prob.premises, prob.hypothesis, kb=kb_raw, context=context)
        )

    # ----- Step 4: Filter KB -----
    kb_results = pipeline_filter_kb_injections(
        kb_raw,
        prob.premises,
        prob.hypothesis,
        filtering=config.filtering,
    )

    if raw_task is not None:
        lp_res_raw = await raw_task
        if lp_res_raw.error:
            log("  [raw-KB] Call failed.")
            exp_result.status_with_raw_kb = ExperimentStepStatus.ERROR
        else:
            exp_result.pred_with_raw_kb = lp_res_raw.label
            exp_result.status_with_raw_kb = ExperimentStepStatus.SUCCESS
            exp_result.prover_calls.append(lp_res_raw)
            log(f"  [raw-KB] Predicted: {lp_res_raw.label.value}")
            
            if lp_res_raw.label == prob.gold_label:
                raw_kb_fixed = True
            log("  [raw-KB] RAW LLM KB solved it.")

        # If only testing raw KB, we're done
        if test_mode == "raw_kb":
            if exp_result.pred_with_raw_kb == prob.gold_label:
                exp_result.final_status = ExperimentStatus.RAW_KB_SOLVED
                exp_result.fixed_by = "raw_kb"
            else:
                exp_result.final_status = ExperimentStatus.RAW_KB_NOT_SOLVED
            if cache_file:
                _save_cache_result(cache_file, exp_result)
            return exp_result

    kb_strings = [res.relation for res in kb_results]
    
    exp_result.kb_filtered = kb_strings
    exp_result.kb_details = kb_results

    log(f"  [KB normalised] {kb_strings}")

    if not kb_strings:
        log("  [KB] All KB removed during normalisation.")
        exp_result.final_status = ExperimentStatus.KB_NORMALISATION_EMPTY
        # If raw KB solved it but normalised KB is empty, raw_kb was the solver.
        if raw_kb_fixed:
            exp_result.fixed_by = "raw_kb"
            exp_result.final_status = ExperimentStatus.RAW_KB_SOLVED
        if cache_file:
            _save_cache_result(cache_file, exp_result)
        return exp_result

    # ----- Step 5: Test with NORMALISED KB -----
    # Check if normalised KB is identical to raw KB
    kb_identical = set(kb_strings) == set(kb_raw)
    if kb_identical and test_mode in ("raw_kb", "both"):
        # Normalised KB is same as raw KB - reuse the raw KB result
        log("  [KB] Normalised KB identical to raw KB - reusing raw KB result.")
        exp_result.pred_with_kb = exp_result.pred_with_raw_kb
        exp_result.status_with_kb = exp_result.status_with_raw_kb
        filtered_kb_fixed = raw_kb_fixed
    else:
        log("  [KB] Calling LangPro with normalised KB...")
        lp_res_with_kb = await langpro_api_call(prob.premises, prob.hypothesis, kb=kb_strings, context=context)
        
        if lp_res_with_kb.error:
            log("  [KB] Call failed.")
            exp_result.status_with_kb = ExperimentStepStatus.ERROR
            exp_result.final_status = ExperimentStatus.NORMALISED_KB_PROVER_FAILED
            if cache_file:
                _save_cache_result(cache_file, exp_result)
            return exp_result

        exp_result.pred_with_kb = lp_res_with_kb.label
        exp_result.status_with_kb = ExperimentStepStatus.SUCCESS
        exp_result.prover_calls.append(lp_res_with_kb)
        filtered_kb_fixed = lp_res_with_kb.label == prob.gold_label
    
    if exp_result.pred_with_kb is not None:
        log(f"  [KB] Predicted: {exp_result.pred_with_kb.value}")
    else:
        log("  [KB] No normalised-KB prediction available.")

    # Determine fixed_by based on what worked
    # Note: "both" only applies when normalisation actually changed the KB
    if filtered_kb_fixed and raw_kb_fixed:
        if kb_identical:
            # Normalisation was a no-op, so credit goes to raw KB only
            exp_result.fixed_by = "raw_kb"
            exp_result.final_status = ExperimentStatus.RAW_KB_SOLVED
            log("  [KB] RAW LLM KB solved it (normalisation made no changes).")
        else:
            exp_result.fixed_by = "both"
            exp_result.final_status = ExperimentStatus.NORMALISED_KB_SOLVED
            log("  [KB] BOTH raw LLM KB and normalised KB solved it.")
    elif filtered_kb_fixed:
        exp_result.fixed_by = "normalised_kb"
        if test_mode in ("raw_kb", "both"):
            log("  [KB] NORMALISED KB solved it (raw LLM KB did not).")
        else:
            log("  [KB] NORMALISED KB solved it (raw KB not tested).")
        exp_result.final_status = ExperimentStatus.NORMALISED_KB_SOLVED
    elif raw_kb_fixed:
        exp_result.fixed_by = "raw_kb"
        log("  [KB] Normalised KB did not solve it, but RAW LLM KB did.")
        exp_result.final_status = ExperimentStatus.RAW_KB_SOLVED
    else:
        log("  [KB] Neither raw nor normalised KB solved it.")
        exp_result.final_status = ExperimentStatus.KB_NOT_SOLVED
        
    # ----- Step 6: Ablation (if requested and something solved it) -----
    if exp_result.fixed_by and config.run_ablation:
        # Run ablation on whichever KB solved it
        if exp_result.fixed_by in ("normalised_kb", "both") and len(kb_strings) > 1:
            log("  [ablation] Running ablation on normalised KB to find minimal sufficient subsets...")
            essential_kb, all_minimals, ablation_results = await _run_ablation(
                prob, kb_strings, prob.gold_label, config.verbose, context=context
            )
            exp_result.essential_kb = essential_kb
            exp_result.ablation_subsets = all_minimals
            exp_result.ablation_results = ablation_results
            log(f"  [ablation] Best minimal subset: {essential_kb}")
            log(f"  [ablation] All minimal subsets ({len(all_minimals)}): {all_minimals}")

    if cache_file:
        _save_cache_result(cache_file, exp_result)

    return exp_result


async def _run_ablation(
    prob: NLIProblem,
    kb_list: List[str],
    gold_label: NLILabel,
    verbose: bool = True,
    context: Optional[AsyncRunContext] = None,
) -> tuple:
    """
    Find all minimal sufficient KB subsets using breadth-first search.
    
    A subset is "minimal sufficient" if:
    1. It produces the correct gold_label when injected
    2. No proper subset of it also produces the correct label
    
    Returns:
        Tuple of (best_minimal_subset, all_minimal_subsets, ablation_log)
        - best_minimal_subset: The minimal subset with lowest token count
        - all_minimal_subsets: All minimal sufficient subsets found
        - ablation_log: Dict mapping tested subsets (as tuple) to resulting label
    """
    from itertools import combinations
    
    minimal_subsets: List[frozenset] = []
    ablation_log = {}
    
    def token_count(subset) -> int:
        """Count total tokens across all KB entries in subset."""
        return sum(len(s.split()) for s in subset)
    
    async def test_subset(subset: List[str]) -> Optional[NLILabel]:
        """Test a KB subset and return the resulting label, or None on error."""
        if not subset:
            # Empty subset - call LangPro with no KB
            res = await langpro_api_call(prob.premises, prob.hypothesis, report=False, context=context)
        else:
            res = await langpro_api_call(prob.premises, prob.hypothesis, kb=subset, context=context)
        
        if res.error:
            return None
        return res.label
    
    # BFS by subset size: test all subsets of size 1, then 2, etc.
    for size in range(1, len(kb_list) + 1):
        if verbose:
            print(f"    [ablation] Testing subsets of size {size}...")
        
        for subset_tuple in combinations(kb_list, size):
            subset_set = frozenset(subset_tuple)
            
            # Skip if this subset is a superset of an already-found minimal
            if any(m.issubset(subset_set) and m != subset_set for m in minimal_subsets):
                if verbose:
                    print(f"    [ablation] Skipping {list(subset_tuple)} (superset of known minimal)")
                continue
            
            # Test this subset
            result_label = await test_subset(list(subset_tuple))
            ablation_log[subset_tuple] = result_label
            
            if result_label is None:
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> ERROR")
                continue
            
            if result_label == gold_label:
                # This subset is sufficient - check if it's minimal
                # (no proper subset also works, but we already checked smaller sizes first)
                minimal_subsets.append(subset_set)
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> {result_label.value} (MINIMAL)")
            else:
                if verbose:
                    print(f"    [ablation] {list(subset_tuple)} -> {result_label.value}")
    
    # Convert frozensets to lists
    all_minimals = [list(s) for s in minimal_subsets]
    
    # Convert tuple keys to strings for serialization (join with |)
    ablation_log_str = {"|".join(k): v for k, v in ablation_log.items()}
    
    # Sort by token count and select best
    if all_minimals:
        sorted_minimals = sorted(all_minimals, key=token_count)
        best = sorted_minimals[0]
    else:
        best = []
    
    return best, all_minimals, ablation_log_str

async def process_kb_examples(
    dataset: DatasetLoader,
    config: Optional[ProblemConfig] = None,
    split: str = "dev",
    label_filter: Set[str] = {"entailment", "contradiction"},
    max_matches: Optional[int] = None, # Renamed conceptual limit, acts as max_yields if needed
    max_checked: Optional[int] = 500,
    problem_ids: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None
) -> AsyncIterator[ExperimentResult]:
    """
    Iterates through dataset examples and yields experiment results for ALL processed problems,
    regardless of outcome.
    
    Args:
        dataset: The dataset loader to use.
        config: Configuration for processing each problem.
        split: Dataset split to use.
        label_filter: Set of gold labels to consider.
        max_matches: Stop after yielding this many results (any outcome).
        max_checked: Maximum number of problems to check/process.
        problem_ids: Optional list of specific problem IDs to process.
        cache_dir: Optional directory for caching results.
    """
    if config is None:
        config = ProblemConfig()

    print(f"[kb-processor] Processing split='{split}', labels={label_filter}")
    print(f"[kb-processor] Config: mode={config.test_mode.value}, ablation={config.run_ablation}")

    checked_count = 0
    yielded_count = 0
    
    used_keys: Set[str] = set()
    
    problem_iterator: Iterator[str]
    is_random = False
    
    if problem_ids:
        print(f"[kb-processor] Using provided list of {len(problem_ids)} problems.")
        problem_iterator = iter(problem_ids)
    else:
        print(f"[kb-processor] Using random sampling.")
        is_random = True
        problem_iterator = iter([])

    while True:
        # Check limits
        if max_matches is not None and yielded_count >= max_matches:
            print(f"[kb-processor] Reached target of {max_matches} results. Stopping.")
            break
        if max_checked is not None and checked_count >= max_checked:
            print(f"[kb-processor] Reached max_checked limit ({max_checked}). Stopping.")
            break
        
        prob: Optional[NLIProblem] = None
        
        if is_random:
            prob = dataset.random_problem(split=split, label_filter=label_filter, exclude_keys=used_keys)
            if not prob:
                print("[kb-processor] No more valid random problems found.")
                break
        else:
            try:
                next_id = next(problem_iterator)
                try:
                    prob = dataset.get_problem(next_id, split=split)
                except KeyError:
                    print(f"[Warning] Problem ID {next_id} not found in dataset.")
                    continue
            except StopIteration:
                print("[kb-processor] Finished processing provided list.")
                break

        if prob.id in used_keys:
            continue
        used_keys.add(prob.id)
        
        cache_file = None
        if cache_dir:
            safe_id = prob.id.replace("/", "_").replace("#", "_")
            fingerprint = result_cache_fingerprint(prob, config)
            filename = f"{prob.dataset}_{split}_{safe_id}_{fingerprint}.json"
            cache_file = cache_dir / filename

            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_result = ExperimentResult.model_validate_json(f.read())
                    if (cached_result.cache_fingerprint != fingerprint
                            or cached_result.resolved_config != config.model_dump(mode="json")
                            or cached_result.filtering_version != FILTERING_PIPELINE_VERSION):
                        raise ValueError("Cached result metadata does not match this configuration")

                    checked_count += 1
                    print(f"\n[kb-processor] #{checked_count} | Key: {prob.id} | Gold: {prob.gold_label.value} [CACHED]")
                    
                    if cached_result.final_status in {
                        ExperimentStatus.NORMALISED_KB_SOLVED,
                        ExperimentStatus.RAW_KB_SOLVED,
                    }:
                        print(f"  [KB] KB solved it ({cached_result.fixed_by})")
                    
                    yielded_count += 1
                    yield cached_result
                    continue

                except Exception as e:
                    print(f"[cache] Error reading {cache_file}: {e}. Reprocessing.")

        checked_count += 1
        print(f"\n[kb-processor] #{checked_count} | Key: {prob.id} | Gold: {prob.gold_label.value}")

        # Delegate to single-problem processor
        exp_result = await process_single_problem(
            prob,
            config=config,
            cache_file=cache_file
        )
        
        yielded_count += 1
        yield exp_result


async def collect_kb_helpful_examples_random(
    dataset: DatasetLoader,
    config: Optional[ProblemConfig] = None,
    split: str = "dev",
    label_filter: Set[str] = {"entailment", "contradiction"},
    max_matches: Optional[int] = 10,
    max_checked: Optional[int] = 500,
    problem_ids: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None
) -> AsyncIterator[ExperimentResult]:
    """
    Wrapper around process_kb_examples that ONLY yields results where KB was helpful
    (i.e., final_status is a solved KB outcome).
    """
    matches_found = 0
    
    # We pass max_matches=None to the inner loop because we want to control the count 
    # based on *matches*, not total processed.
    generator = process_kb_examples(
        dataset=dataset,
        config=config,
        split=split,
        label_filter=label_filter,
        max_matches=None, 
        max_checked=max_checked,
        problem_ids=problem_ids,
        cache_dir=cache_dir
    )
    
    async for result in generator:
        if result.final_status in {
            ExperimentStatus.NORMALISED_KB_SOLVED,
            ExperimentStatus.RAW_KB_SOLVED,
        }:
            matches_found += 1
            yield result
            
            if max_matches is not None and matches_found >= max_matches:
                print(f"[kb-collector] Reached target of {max_matches} helpful matches. Stopping.")
                break
                
    print(f"\n[kb-collector] Finished. Found {matches_found} KB-helpful examples.")
