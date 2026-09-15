"""Load and fill the three pipeline prompt templates from <repo root>/prompts/."""

from __future__ import annotations

import hashlib
import re

from .paths import PROMPTS_DIR

PROMPT_FILES = {
    "lex": "knowledge_generation.txt",
    "stefan": "knowledge_generation_stefan.txt",
}
CRITIC_PROMPT_FILE = "failure_analysis.txt"
RETRY_PROMPT_FILE = "knowledge_refinement.txt"


def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def prompt_text(profile: str = "lex") -> str:
    try:
        filename = PROMPT_FILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt profile {profile!r}; choose from {sorted(PROMPT_FILES)}") from exc
    return _read(filename)


def prompt_sha256(profile: str = "lex") -> str:
    return hashlib.sha256(prompt_text(profile).encode("utf-8")).hexdigest()


def _rules_block(profile: str = "lex") -> str:
    """All shared initial-prompt instructions and examples for a retry."""
    full = prompt_text(profile)
    markers = list(re.finditer(r"(?im)^\s*premise:\s*\{premise\}\s*$", full))
    if not markers:
        raise ValueError(f"{PROMPT_FILES[profile]} must contain a Premise: {{premise}} line")
    end = markers[-1].start()
    return full[:end].strip()


def fill_kb_prompt(premise: str, hypothesis: str, *, profile: str = "lex") -> str:
    template = prompt_text(profile)
    return template.format(premise=premise, hypothesis=hypothesis)


def fill_retry_prompt(
    premise: str,
    hypothesis: str,
    *,
    analysis: str,
    previous_kb: str,
    kb_label: str,
    profile: str = "lex",
) -> str:
    template = _read(RETRY_PROMPT_FILE)
    return template.format(
        base_rules=_rules_block(profile),
        analysis=analysis.strip(),
        previous_kb=previous_kb.strip() or "(empty)",
        kb_label=kb_label,
        premise=premise,
        hypothesis=hypothesis,
    )


def fill_critic_prompt(
    premise: str,
    hypothesis: str,
    *,
    baseline_label: str,
    attempted_kb: str,
    kb_label: str,
    closure_pattern: str = "",
    proof_info: str = "",
    kb_verification: str = "",
    tableau_comparison: str = "",
    previous_notes: str = "",
) -> str:
    template = _read(CRITIC_PROMPT_FILE)
    return template.format(
        premise=premise,
        hypothesis=hypothesis,
        baseline_label=baseline_label,
        attempted_kb=attempted_kb.strip() or "(empty)",
        kb_label=kb_label,
        closure_pattern=closure_pattern.strip() or "(not available)",
        proof_info=proof_info.strip() or "(not available)",
        kb_verification=kb_verification.strip() or "(not available)",
        tableau_comparison=tableau_comparison.strip() or "(not available)",
        previous_notes=previous_notes.strip() or "(none)",
    )
