import re
from functools import lru_cache
import os
import warnings
from typing import Optional, Tuple, List, Set, Union
import nltk
from .downloads import check_nltk
from nltk.stem import WordNetLemmatizer
from .models import KBResult, FilteringConfig
from .filtering_context import FilteringContext, prepare_filtering_resources

FILTERING_PIPELINE_VERSION = "2.0-pos-configurable"


# =============================================================================
# Cached model loaders
# =============================================================================

@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    check_nltk('wordnet')
    return WordNetLemmatizer()


@lru_cache(maxsize=4)
def _get_st_model_cached(model_name: str, cache_folder: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    if cache_folder:
        return SentenceTransformer(model_name, cache_folder=cache_folder)
    return SentenceTransformer(model_name)


def get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    return _get_st_model_cached(model_name, cache_folder)


# =============================================================================
# Helper functions
# =============================================================================

def tokenize(text: str) -> List[str]:
    check_nltk('punkt_tab')
    return [t.lower() for t in nltk.word_tokenize(text)]


def remove_underscores(s: str) -> str:
    """Replace underscores with spaces."""
    return s.replace("_", " ")


def normalize_kb_args(kb: str) -> str:
    pred, a, b = parse_kb_injection(kb)
    a = remove_underscores(a)
    b = remove_underscores(b)
    return f"{pred}({a}, {b})"


def drop_leading_preposition(phrase: str) -> str:
    """
    If the phrase has exactly 3 tokens and the first is a preposition (IN),
    drop the first token. Otherwise, return the phrase unchanged.
    """
    check_nltk('punkt_tab')
    check_nltk('averaged_perceptron_tagger_eng')
    tokens = nltk.word_tokenize(phrase)

    if len(tokens) == 3:
        tagged = nltk.pos_tag(tokens)
        word, tag = tagged[0]

        if tag == "IN":   # preposition
            return " ".join(tokens[1:])

    return phrase


def parse_kb_injection(kb: str) -> Tuple[str, str, str]:
    """
    Parse predicate(arg1, arg2) into (predicate, arg1, arg2)
    Works for isa_wn, disj, and any other binary predicate.
    """
    m = re.fullmatch(r'\s*(\w+)\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*', kb)
    if not m:
        raise ValueError(f"Invalid KB format: {kb}")
    return m.group(1), m.group(2), m.group(3)


def create_rel(pred: str, arg1: str, arg2: str) -> str:
    return f"{pred}({arg1}, {arg2})"


def normalize_premises(premises: Union[str, List[str]]) -> str:
    """
    Normalize premise input to a single concatenated string.
    Accepts either a single string or a list of strings.
    """
    if isinstance(premises, list):
        return " ".join(premises)
    return premises


ALLOWED_PREDICATES = {"isa_wn", "disj"}


# =============================================================================
# Phase 1: Candidate Generation
# =============================================================================

def _record(audit, stage, before, after, reason=None):
    if audit is not None:
        event = {"stage": stage, "before": before, "after": after}
        if reason is not None:
            event["reason"] = reason
        audit.append(event)


def _variant(candidate, arg1, arg2, stage, provenance, audit=None, reason=None):
    pred, _, _ = parse_kb_injection(candidate.relation)
    relation = create_rel(pred, arg1, arg2)
    _record(audit, stage, candidate.relation, relation, reason)
    return candidate.model_copy(update={
        "relation": relation,
        "provenance": provenance,
        "transformations": candidate.transformations + [stage],
        "alignment_reason": reason if reason is not None else candidate.alignment_reason,
    })


def _deduplicate(candidates, audit=None):
    seen = set()
    output = []
    for candidate in candidates:
        if candidate.relation in seen:
            _record(audit, "deduplication", candidate.relation, None, "duplicate")
            continue
        seen.add(candidate.relation)
        output.append(candidate)
    return output


def _apply_mode(candidates, mode, transform):
    if mode == "off":
        return candidates
    output = []
    for candidate in candidates:
        transformed = transform(candidate)
        if mode == "additive":
            output.append(candidate)
        if transformed is not None:
            output.append(transformed)
        elif mode == "replacement":
            output.append(candidate)
    return output


def _resolve_filtering(filtering, post_process=None, use_semantic=None):
    config = filtering if filtering is not None else FilteringConfig.operational()
    changes = {}
    if post_process is not None:
        mode = "replacement" if post_process else "off"
        if filtering is not None and (
            config.underscores_mode != mode or config.leading_preposition_mode != mode
        ):
            raise ValueError("post_process conflicts with the explicit filtering configuration")
        changes.update(underscores_mode=mode, leading_preposition_mode=mode)
    if use_semantic is not None:
        if filtering is not None and config.use_semantic != use_semantic:
            raise ValueError("use_semantic conflicts with the explicit filtering configuration")
        changes["use_semantic"] = use_semantic
    return FilteringConfig(**(config.model_dump() | changes))


def generate_all_candidates(
    pred: str,
    arg1: str,
    arg2: str,
    original_text: str,
    post_process: Optional[bool] = None,
    *,
    filtering: Optional[FilteringConfig] = None,
    context: Optional[FilteringContext] = None,
    premise: Union[str, List[str]] = "",
    hypothesis: str = "",
    audit: Optional[List[dict]] = None,
) -> List[KBResult]:
    """Generate configured variants, preserving the historical additive DAG.

    Lemma and diff additions are siblings of the normalized candidate. If either
    stage uses replacement, they operate sequentially (lemma then diff), so a
    replaced ancestor cannot reappear. Alignment replacement runs before lemma;
    additive alignment runs after lemma/diff, as in the historical pipeline.
    Contextual POS without supplied sentence context leaves arguments unchanged.
    """
    config = _resolve_filtering(filtering, post_process)
    context = context or FilteringContext(premise, hypothesis)
    candidates = [KBResult(relation=create_rel(pred, arg1, arg2),
                           provenance="llm", original_text=original_text)]

    def normalization(candidate, operation, stage):
        _, a, b = parse_kb_injection(candidate.relation)
        new_a, new_b = operation(a), operation(b)
        if (new_a, new_b) == (a, b):
            return None
        return _variant(candidate, new_a, new_b, stage, "post_process", audit)

    candidates = _apply_mode(candidates, config.underscores_mode,
        lambda c: normalization(c, remove_underscores, "underscores"))
    candidates = _deduplicate(candidates, audit)
    candidates = _apply_mode(candidates, config.leading_preposition_mode,
        lambda c: normalization(c, context.drop_preposition, "leading_preposition"))
    candidates = _deduplicate(candidates, audit)

    if config.argument_alignment_mode == "replacement":
        aligned = []
        for candidate in candidates:
            _, a, b = parse_kb_injection(candidate.relation)
            direct = (context.matches(a, context.premise, config.lemma_match_policy)
                      and context.matches(b, context.hypothesis, config.lemma_match_policy))
            reverse = (context.matches(b, context.premise, config.lemma_match_policy)
                       and context.matches(a, context.hypothesis, config.lemma_match_policy))
            reason = ("ambiguous" if direct and reverse else "already_aligned" if direct
                      else "reoriented" if reverse else "unaligned")
            if reason == "reoriented":
                prov = (candidate.provenance + "_swap"
                        if candidate.provenance != "llm" else "derived_swap")
                candidate = _variant(candidate, b, a, "argument_alignment", prov, audit, reason)
            else:
                _record(audit, "argument_alignment", candidate.relation,
                        candidate.relation, reason)
                candidate = candidate.model_copy(update={"alignment_reason": reason})
            aligned.append(candidate)
        candidates = aligned

    def lemma_variant(candidate):
        _, a, b = parse_kb_injection(candidate.relation)
        la = context.lemmatize_argument(a, context.premise, config.lemmatization_kind)
        lb = context.lemmatize_argument(b, context.hypothesis, config.lemmatization_kind)
        # Historical verb mode did not emit a casing-only variant.
        unchanged = (la, lb) == ((a.lower(), b.lower()) if config.lemmatization_kind == "verb" else (a, b))
        if unchanged:
            return None
        return _variant(candidate, la, lb, "lemmatization_" + config.lemmatization_kind,
                        "derived_lemma", audit)

    def diff_variant(candidate):
        _, a, b = parse_kb_injection(candidate.relation)
        left, right = a.split(), b.split()
        if len(left) != len(right) or len(left) <= 1:
            return None
        differences = [(x, y) for x, y in zip(left, right) if x.lower() != y.lower()]
        if not differences:
            return None
        da, db = (" ".join(pair[i] for pair in differences) for i in (0, 1))
        if (da, db) == (a, b):
            return None
        return _variant(candidate, da, db, "diff_only", "derived_diff", audit)

    if "replacement" in (config.lemmatization_mode, config.diff_only_mode):
        candidates = _apply_mode(candidates, config.lemmatization_mode, lemma_variant)
        candidates = _apply_mode(candidates, config.diff_only_mode, diff_variant)
    else:
        siblings = []
        for candidate in candidates:
            siblings.append(candidate)
            if config.lemmatization_mode == "additive":
                lemma = lemma_variant(candidate)
                if lemma is not None:
                    siblings.append(lemma)
            if config.diff_only_mode == "additive":
                diff = diff_variant(candidate)
                if diff is not None:
                    siblings.append(diff)
        candidates = siblings

    output = []
    for candidate in candidates:
        _, a, b = parse_kb_injection(candidate.relation)
        if a.lower() == b.lower():
            _record(audit, "acceptance", candidate.relation, None, "identical_arguments")
            continue
        output.append(candidate)
        if config.argument_alignment_mode == "additive":
            prov = candidate.provenance + "_swap" if candidate.provenance != "llm" else "derived_swap"
            output.append(_variant(candidate, b, a, "argument_alignment", prov, audit))
    return _deduplicate(output, audit)


# =============================================================================
# Phase 2: Filtering
# =============================================================================

def is_arg_in_text(
    arg_str: str,
    tokens: List[str],
    *,
    lemma_match_policy: str = "exact_or_lemma",
) -> bool:
    """Match all argument components using same-POS noun/verb lemma indices.

    The main pipeline reuses a FilteringContext instead of constructing one per
    call. This public compatibility helper accepts the historical token list.
    """
    context = FilteringContext("", "")
    text = " ".join(tokens)
    return context.matches(arg_str, text, lemma_match_policy)


def is_arg_semantically_similar(
    arg_str: str,
    tokens: List[str],
    st_model=None,
    threshold: float = 0.60,
    use_token_level: bool = True
) -> bool:
    """
    Check if an argument is semantically similar to text tokens.
    
    This is a SOFTER check used for final validation, not for determining
    which sentence contains the argument.
    """
    lemmatizer = get_lemmatizer()
    text_str = " ".join(tokens).lower()
    arg_str = (arg_str or "").strip().lower()

    if not arg_str:
        return False

    # Fallback if no model
    if st_model is None:
        return arg_str in text_str

    # For multi-word args, check each part
    parts = tokenize(arg_str)
    if not parts:
        return False

    token_set = {t.lower() for t in tokens}
    token_lemmas = {lemmatizer.lemmatize(t.lower()) for t in tokens}
    token_lemmas_v = {lemmatizer.lemmatize(t.lower(), pos='v') for t in tokens}

    for part in parts:
        part_l = part.lower()

        # Fast checks first
        if part_l in token_set:
            continue
        if lemmatizer.lemmatize(part_l) in token_lemmas:
            continue
        if lemmatizer.lemmatize(part_l, pos='v') in token_lemmas_v:
            continue

        # Semantic soft match
        if use_token_level:
            cand_tokens = list({t.lower() for t in tokens if t})
            if not cand_tokens:
                return False

            emb = st_model.encode([part_l] + cand_tokens, convert_to_tensor=True)
            part_emb = emb[0]
            token_embs = emb[1:]

            from sentence_transformers import util
            sims = util.cos_sim(part_emb, token_embs)
            best_score = float(sims.max().item())

            if best_score < threshold:
                return False
        else:
            emb = st_model.encode([arg_str, text_str], convert_to_tensor=True)
            from sentence_transformers import util
            score = float(util.cos_sim(emb[0], emb[1]).item())
            if score < threshold:
                return False

    return True


def filter_candidates(
    candidates: List[KBResult],
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    threshold: float = 0.60,
    strict: bool = True,
    use_semantic: bool = False,
    *,
    lemma_match_policy: str = "exact_or_lemma",
    context: Optional[FilteringContext] = None,
    audit: Optional[List[dict]] = None,
) -> List[KBResult]:
    """Filter already-generated candidates using reusable P/H matching indices."""
    context = context or FilteringContext(premise, hypothesis)
    if use_semantic and context.offline:
        raise ValueError("Offline filtering does not support semantic model loading.")
    if use_semantic and st_model is None:
        st_model = get_st_model()
    left = context.premise if strict else context.premise + " " + context.hypothesis
    right = context.hypothesis if strict else left
    output = []
    for candidate in candidates:
        try:
            _, a, b = context.parse(candidate.relation)
        except ValueError:
            _record(audit, "final_ph_filter", candidate.relation, None, "invalid_syntax")
            continue
        if use_semantic:
            accepted = (
                is_arg_semantically_similar(a, list(context.tokens(left)), st_model, threshold)
                and is_arg_semantically_similar(b, list(context.tokens(right)), st_model, threshold)
            )
        else:
            accepted = (context.matches(a, left, lemma_match_policy)
                        and context.matches(b, right, lemma_match_policy))
        if accepted:
            output.append(candidate)
        else:
            _record(audit, "final_ph_filter", candidate.relation, None, "ph_mismatch")
    return _deduplicate(output, audit)


def pipeline_filter_kb_injections(
    kb_list: List[str],
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    post_process: Optional[bool] = None,
    use_semantic: Optional[bool] = None,
    *,
    filtering: Optional[FilteringConfig] = None,
    context: Optional[FilteringContext] = None,
    offline: bool = False,
    audit: Optional[List[dict]] = None,
) -> List[KBResult]:
    """Transform and optionally filter KB relations using one resolved config.

    The default profile uses POS additive lemmatization. Explicit ``filtering``
    settings also drive offline F1 replay; the scorer never changes direction
    equivalence. Pass a reusable context to share NLP work across configurations.
    Legacy ``post_process`` controls only the two textual normalizations and
    raises on conflict with explicit settings. Audit events record transformations,
    alignment decisions, rejection reasons and stable deduplication.
    """
    config = _resolve_filtering(filtering, post_process, use_semantic)
    if context is None:
        context = FilteringContext(premise, hypothesis, offline=offline)
    elif context.premise != normalize_premises(premise) or context.hypothesis != hypothesis:
        raise ValueError("FilteringContext belongs to a different premise/hypothesis pair")
    elif offline and not context.offline:
        raise ValueError("offline=True requires an offline FilteringContext")
    if context.offline and config.use_semantic:
        raise ValueError("Offline filtering does not support semantic model loading.")
    candidates = []
    for injection in kb_list:
        try:
            pred, a, b = context.parse(injection)
        except ValueError:
            _record(audit, "acceptance", injection, None, "invalid_syntax")
            continue
        if pred not in ALLOWED_PREDICATES:
            _record(audit, "acceptance", injection, None, "unsupported_predicate")
            continue
        candidates.extend(generate_all_candidates(
            pred, a, b, injection, filtering=config, context=context, audit=audit,
        ))
    if config.final_ph_filter:
        return filter_candidates(candidates, premise, hypothesis, st_model=st_model,
            use_semantic=config.use_semantic, lemma_match_policy=config.lemma_match_policy,
            context=context, audit=audit)
    return _deduplicate(candidates, audit)


# =============================================================================
# Legacy function (for backwards compatibility)
# =============================================================================

def filter_kb_by_prem_hyp(
    kb_list,
    premise: Union[str, List[str]],
    hypothesis: str,
    st_model=None,
    threshold: float = 0.60,
    swap_args: Optional[bool] = None,
    strict: bool = True,
    use_token_level: bool = True,
) -> List[KBResult]:
    """
    Legacy function - now just wraps filter_candidates.
    
    Filters KB injections using exact/lemma matching.
    """
    if swap_args is not None:
        warnings.warn("swap_args never changed this legacy filtering wrapper; use "
                      "pipeline_filter_kb_injections with argument_alignment_mode instead.",
                      DeprecationWarning, stacklevel=2)
    # Convert string inputs to KBResult if needed
    candidates: List[KBResult] = []
    for kb_item in kb_list:
        if isinstance(kb_item, str):
            candidates.append(KBResult(relation=kb_item, provenance="llm", original_text=kb_item))
        else:
            candidates.append(kb_item)
    
    return filter_candidates(
        candidates,
        premise,
        hypothesis,
        st_model=st_model,
        threshold=threshold,
        strict=strict,
        use_semantic=False  # Use strict matching
    )
