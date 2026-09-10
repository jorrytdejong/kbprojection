import asyncio
import json
import os
from typing import Any, Optional, Sequence

from .async_runtime import AsyncRunContext, AsyncRunLimits, create_async_run_context
from .models import ExperimentResult, LangProResult, NLIProblem, ProblemConfig, TestMode
from .orchestration import experiment_result_to_json, process_single_problem


def infer_provider(model: str, explicit_provider: Optional[str] = None) -> str:
    if explicit_provider:
        return explicit_provider
    if "/" in model and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if "/" not in model and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise ValueError("Could not infer provider. Set a provider API key or pass provider explicitly.")


def serialize_result_payload(
    result: ExperimentResult,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt_style: Optional[str] = None,
    discard_prover_calls: bool = False,
) -> dict[str, Any]:
    payload = json.loads(experiment_result_to_json(result))
    if discard_prover_calls:
        payload["prover_calls"] = None
    if model is not None:
        payload["model"] = model
    if provider is not None:
        payload["provider"] = provider
    if prompt_style is not None:
        payload["prompt_style"] = prompt_style
    return payload


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _build_config(
    *,
    model: str,
    provider: Optional[str],
    prompt_style: str,
    test_mode: str | TestMode,
    post_process: Optional[bool],
    run_ablation: bool,
    verbose: bool,
    config: Optional[ProblemConfig],
) -> ProblemConfig:
    if config is not None:
        if post_process is not None:
            raise ValueError("Pass cleanup settings in config.filtering when config is supplied; do not also pass post_process")
        return config

    return ProblemConfig(
        llm_provider=infer_provider(model, provider),
        model=model,
        prompt_style=prompt_style,
        test_mode=TestMode(test_mode),
        post_process=post_process,
        run_ablation=run_ablation,
        verbose=verbose,
    )


def _build_context(
    *,
    llm_concurrency: int,
    langpro_concurrency: int,
    local_langpro_concurrency: int,
    context: Optional[AsyncRunContext],
) -> AsyncRunContext:
    if context is not None:
        return context

    _validate_positive("llm_concurrency", llm_concurrency)
    _validate_positive("langpro_concurrency", langpro_concurrency)
    _validate_positive("local_langpro_concurrency", local_langpro_concurrency)
    return create_async_run_context(
        AsyncRunLimits(
            llm_concurrency=llm_concurrency,
            langpro_concurrency=langpro_concurrency,
            local_langpro_concurrency=local_langpro_concurrency,
        )
    )


async def arun_problem(
    problem: NLIProblem,
    *,
    model: str = "gpt-5-mini",
    provider: Optional[str] = None,
    prompt_style: str = "icl",
    test_mode: str | TestMode = TestMode.BOTH,
    post_process: Optional[bool] = None,
    run_ablation: bool = False,
    verbose: bool = True,
    llm_concurrency: int = 2,
    langpro_concurrency: int = 4,
    local_langpro_concurrency: int = 2,
    context: Optional[AsyncRunContext] = None,
    config: Optional[ProblemConfig] = None,
    baseline_no_kb: Optional[LangProResult] = None,
) -> ExperimentResult:
    resolved_config = _build_config(
        model=model,
        provider=provider,
        prompt_style=prompt_style,
        test_mode=test_mode,
        post_process=post_process,
        run_ablation=run_ablation,
        verbose=verbose,
        config=config,
    )
    resolved_context = _build_context(
        llm_concurrency=llm_concurrency,
        langpro_concurrency=langpro_concurrency,
        local_langpro_concurrency=local_langpro_concurrency,
        context=context,
    )
    return await process_single_problem(
        problem,
        config=resolved_config,
        context=resolved_context,
        baseline_no_kb=baseline_no_kb,
    )


def run_problem(problem: NLIProblem, **kwargs: Any) -> ExperimentResult:
    return asyncio.run(arun_problem(problem, **kwargs))


async def arun_problems(
    problems: Sequence[NLIProblem],
    *,
    model: str = "gpt-5-mini",
    provider: Optional[str] = None,
    prompt_style: str = "icl",
    test_mode: str | TestMode = TestMode.BOTH,
    post_process: Optional[bool] = None,
    run_ablation: bool = False,
    verbose: bool = True,
    concurrency: int = 4,
    llm_concurrency: int = 2,
    langpro_concurrency: int = 4,
    local_langpro_concurrency: int = 2,
    discard_prover_calls: bool = False,
    context: Optional[AsyncRunContext] = None,
    config: Optional[ProblemConfig] = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    _validate_positive("concurrency", concurrency)
    problem_list = list(problems)
    if not problem_list:
        return []

    resolved_config = _build_config(
        model=model,
        provider=provider,
        prompt_style=prompt_style,
        test_mode=test_mode,
        post_process=post_process,
        run_ablation=run_ablation,
        verbose=verbose,
        config=config,
    )
    resolved_context = _build_context(
        llm_concurrency=llm_concurrency,
        langpro_concurrency=langpro_concurrency,
        local_langpro_concurrency=local_langpro_concurrency,
        context=context,
    )
    job_semaphore = asyncio.Semaphore(concurrency)

    async def process_indexed(index: int, problem: NLIProblem) -> tuple[int, dict[str, Any]]:
        async with job_semaphore:
            result = await process_single_problem(
                problem,
                config=resolved_config,
                context=resolved_context,
            )
        payload = serialize_result_payload(
            result,
            model=resolved_config.model,
            provider=resolved_config.llm_provider,
            prompt_style=resolved_config.prompt_style,
            discard_prover_calls=discard_prover_calls,
        )
        return index, payload

    tasks = [
        asyncio.create_task(process_indexed(index, problem))
        for index, problem in enumerate(problem_list)
    ]
    ordered: list[Optional[dict[str, Any]]] = [None] * len(problem_list)

    progress = None
    iterator: Any = asyncio.as_completed(tasks)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(iterator, total=len(tasks), unit="problem")
            iterator = progress
        except ImportError:
            pass

    try:
        for completed in iterator:
            index, payload = await completed
            ordered[index] = payload
    except Exception:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        if progress is not None:
            progress.close()

    return [payload for payload in ordered if payload is not None]


def run_problems(problems: Sequence[NLIProblem], **kwargs: Any) -> list[dict[str, Any]]:
    return asyncio.run(arun_problems(problems, **kwargs))
