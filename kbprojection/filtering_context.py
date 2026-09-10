"""Lazy, reusable NLP state for filtering, including a fail-closed offline mode."""

from functools import lru_cache
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import nltk
from nltk.stem import WordNetLemmatizer

from .downloads import check_nltk


RESOURCE_PATHS = {
    "punkt_tab": ("tokenizers/punkt_tab/english/",),
    "averaged_perceptron_tagger_eng": ("taggers/averaged_perceptron_tagger_eng/",),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
}


def _find_resource(name: str) -> str:
    for path in RESOURCE_PATHS[name]:
        try:
            return str(nltk.data.find(path))
        except LookupError:
            pass
    raise LookupError(
        f"Offline filtering requires the installed NLTK resource {name!r}. "
        f"Install it once with: python -m nltk.downloader {name}. "
        "No download was attempted."
    )


def prepare_filtering_resources(configs: Iterable, nltk_data: Optional[str] = None) -> dict:
    """Verify resources required by resolved configurations without downloading.

    The returned paths can be recorded in the replay manifest. The optional data
    directory is registered with NLTK before lookup; unrelated resources are not
    required by disabled transformations. Call this before using NLP in a fresh
    process: NLTK itself caches loaded models, so changing data paths after loading
    models does not reliably change their resources. Lookup uses NLTK's normal
    fallback paths; the returned paths identify where each resource was found.
    """
    # Discover only conventional locations; never depend on a personal dossier.
    root = Path(os.environ.get("KBPROJECTION_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    cache = Path(os.environ.get("KBPROJECTION_THIRD_PARTY_CACHE", root / ".cache"))
    candidates = [cache / "nltk", root / "nltk_data", Path(sys.prefix).parent / "nltk_data"]
    for path in candidates:
        if path.is_dir() and str(path) not in nltk.data.path:
            nltk.data.path.append(str(path))
    # NLTK normally reads this at import; also honor an explicitly configured runtime.
    overrides = ([str(nltk_data)] if nltk_data is not None else
                 [path for path in os.environ.get("NLTK_DATA", "").split(os.pathsep) if path])
    for path in reversed(overrides):
        if path in nltk.data.path:
            nltk.data.path.remove(path)
        nltk.data.path.insert(0, path)
    required = set()
    for config in configs:
        if config.use_semantic:
            raise ValueError("Offline replay does not support semantic model loading.")
        if config.leading_preposition_mode != "off":
            required.update(("punkt_tab", "averaged_perceptron_tagger_eng"))
        if config.lemmatization_mode != "off":
            required.update(("punkt_tab", "wordnet"))
            if config.lemmatization_kind == "pos":
                required.add("averaged_perceptron_tagger_eng")
        if config.final_ph_filter or config.argument_alignment_mode == "replacement":
            required.add("punkt_tab")
            if config.lemma_match_policy != "exact":
                required.add("wordnet")
    return {name: _find_resource(name) for name in sorted(required)}


class FilteringContext:
    """Reusable state for one P/H pair, safe to share across filtering settings.

    Tokens, sentence POS tags, argument lemmas and per-sentence matching indices
    are cached separately. Nothing is loaded until a transformation needs it.
    Multiword matching intentionally retains the historical all-components
    semantics; contextual POS requires a contiguous, unambiguous token span.
    """

    def __init__(self, premise, hypothesis: str, *, offline: bool = False):
        self.premise = " ".join(premise) if isinstance(premise, list) else premise
        self.hypothesis = hypothesis
        self.offline = offline
        self._resources = set()
        self._lemmatizer = None
        # Bound caches to the context lifetime, not a process-global method cache.
        self.tokens = lru_cache(maxsize=None)(self._tokens)
        self.tags = lru_cache(maxsize=None)(self._tags)
        self.lemma = lru_cache(maxsize=None)(self._lemma)
        self.index = lru_cache(maxsize=None)(self._index)
        self.matches = lru_cache(maxsize=None)(self._matches)
        self.drop_preposition = lru_cache(maxsize=None)(self._drop_preposition)
        self.parse = lru_cache(maxsize=None)(self._parse)

    def require(self, name: str):
        if name not in self._resources:
            if self.offline:
                _find_resource(name)
            else:
                check_nltk(name)
            self._resources.add(name)

    def _tokens(self, text: str):
        self.require("punkt_tab")
        return tuple(nltk.word_tokenize(text))

    def _tags(self, text: str):
        self.require("averaged_perceptron_tagger_eng")
        return tuple(nltk.pos_tag(self.tokens(text)))

    def _lemma(self, token: str, pos: str):
        if self._lemmatizer is None:
            self.require("wordnet")
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer.lemmatize(token.lower(), pos=pos)

    def _index(self, text: str, policy: str):
        tokens = frozenset(token.lower() for token in self.tokens(text))
        if policy == "exact":
            return tokens, frozenset(), frozenset()
        return (tokens, frozenset(self.lemma(t, "n") for t in tokens),
                frozenset(self.lemma(t, "v") for t in tokens))

    def _matches(self, argument: str, text: str, policy: str):
        if policy not in {"exact", "exact_or_lemma", "lemma_only"}:
            raise ValueError(f"Unknown lemma match policy: {policy}")
        parts = tuple(t.lower() for t in self.tokens(argument.strip()))
        if not parts:
            return False
        exact, nouns, verbs = self.index(text, policy)
        for part in parts:
            if policy != "lemma_only" and part in exact:
                continue
            if policy != "exact" and (
                self.lemma(part, "n") in nouns or self.lemma(part, "v") in verbs
            ):
                continue
            return False
        return True

    def _drop_preposition(self, text: str):
        tokens = self.tokens(text)
        if len(tokens) == 3 and self.tags(text)[0][1] == "IN":
            return " ".join(tokens[1:])
        return text

    @staticmethod
    def _parse(text: str):
        # Lazy import keeps the public parser and this cache on one definition.
        from .filtering import parse_kb_injection
        return parse_kb_injection(text)

    def lemmatize_argument(self, text: str, sentence: str, kind: str) -> str:
        lower = tuple(token.lower() for token in self.tokens(text))
        if kind == "verb":
            return " ".join(self.lemma(token, "v") for token in lower)
        if kind != "pos":
            raise ValueError(f"Unknown lemmatization kind: {kind}")
        tagged = self.tags(sentence)
        words = tuple(word.lower() for word, _ in tagged)
        matches = {
            tuple(tag for _, tag in tagged[i:i + len(lower)])
            for i in range(len(tagged) - len(lower) + 1)
            if words[i:i + len(lower)] == lower
        }
        if not lower or len(matches) != 1:
            return text
        mapping = {"J": "a", "V": "v", "N": "n", "R": "r"}
        return " ".join(
            self.lemma(token, mapping[tag[0]]) if tag[0] in mapping else token
            for token, tag in zip(lower, next(iter(matches)))
        )
