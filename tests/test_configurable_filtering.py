"""Filtering contracts: actual directed alignment and shared transformation modes."""

import re
import unittest
from unittest.mock import patch

from kbprojection.filtering import (
    FilteringContext, filter_kb_by_prem_hyp, generate_all_candidates,
    pipeline_filter_kb_injections, prepare_filtering_resources,
)
from kbprojection.models import FilteringConfig


class SmallContext(FilteringContext):
    """Deterministic NLP fixture; the actual matching/POS algorithms stay intact."""

    LEMMAS = {("children", "n"): "child", ("cats", "n"): "cat",
              ("dogs", "n"): "dog", ("running", "v"): "run",
              ("cleaning", "v"): "clean", ("tidying", "v"): "tidy",
              ("smiling", "v"): "smile", ("better", "a"): "good",
              ("faster", "r"): "fast", ("saw", "v"): "see"}

    def require(self, name):
        raise AssertionError("Fixture unexpectedly loaded a resource")

    def _tokens(self, text):
        return tuple(re.findall(r"\w+|[^\w\s]", text))

    def _tags(self, text):
        tags = {"children": "NNS", "cats": "NNS", "dogs": "NNS", "running": "VBG",
                "cleaning": "VBG", "tidying": "VBG", "smiling": "VBG", "better": "JJR",
                "faster": "RBR", "in": "IN", "up": "RP", "is": "VBZ"}
        tagged = []
        for token in self.tokens(text):
            # A deliberately ambiguous repeated span tests conservative POS.
            tag = ("NN" if not tagged else "VBD") if token == "saw" else tags.get(token.lower(), "NN")
            tagged.append((token, tag))
        return tuple(tagged)

    def _lemma(self, token, pos):
        return self.LEMMAS.get((token.lower(), pos), token.lower())


def config(**kwargs):
    return FilteringConfig.evaluation_baseline(**kwargs)


def run(kb, premise, hypothesis, filtering, audit=None, context=None):
    return pipeline_filter_kb_injections(kb, premise, hypothesis, filtering=filtering,
        context=context or SmallContext(premise, hypothesis), audit=audit)


class ConfigurableFilteringTests(unittest.TestCase):
    def test_two_profiles_have_intentional_different_lemma_defaults(self):
        self.assertEqual(FilteringConfig.operational().lemmatization_kind, "pos")
        self.assertEqual(FilteringConfig.operational().lemmatization_mode, "additive")
        self.assertEqual(config().lemmatization_mode, "off")
        self.assertFalse(config().final_ph_filter)

    def test_lemma_modes_preserve_or_replace_nouns(self):
        for mode, expected in (
            ("off", ["isa_wn(children, people)"]),
            ("additive", ["isa_wn(children, people)", "isa_wn(child, people)"]),
            ("replacement", ["isa_wn(child, people)"]),
        ):
            with self.subTest(mode=mode):
                result = run(["isa_wn(children, people)"], "children", "people",
                             config(lemmatization_mode=mode))
                self.assertEqual([r.relation for r in result], expected)

    def test_pos_uses_noun_verb_adjective_and_adverb_tags(self):
        ctx = SmallContext("children running better faster", "")
        self.assertEqual(ctx.lemmatize_argument("children running better faster", ctx.premise, "pos"),
                         "child run good fast")
        self.assertEqual(ctx.lemmatize_argument("children", ctx.premise, "verb"), "children")

    def test_absent_and_ambiguous_pos_spans_remain_unchanged(self):
        ctx = SmallContext("saw a saw", "")
        self.assertEqual(ctx.lemmatize_argument("saw", ctx.premise, "pos"), "saw")
        self.assertEqual(ctx.lemmatize_argument("running", ctx.premise, "pos"), "running")

    def test_emma_reorients_actual_pair_without_equating_arguments(self):
        events = []
        result = run(["isa_wn(tidy up, clean)"], "Emma is cleaning the room",
                     "Emma is tidying up the room", config(argument_alignment_mode="replacement"), events)
        self.assertEqual([r.relation for r in result], ["isa_wn(clean, tidy up)"])
        self.assertEqual(result[0].alignment_reason, "reoriented")
        self.assertEqual(result[0].provenance, "derived_swap")
        self.assertEqual(events[0]["before"], "isa_wn(tidy up, clean)")

    def test_alignment_keeps_direct_ambiguous_and_neither(self):
        for premise, hypothesis, reason in (
            ("dog", "animal", "already_aligned"),
            ("dog animal", "dog animal", "ambiguous"),
            ("cat", "bird", "unaligned"),
        ):
            with self.subTest(reason=reason):
                result = run(["isa_wn(dog, animal)"], premise, hypothesis,
                             config(argument_alignment_mode="replacement"))
                self.assertEqual(result[0].relation, "isa_wn(dog, animal)")
                self.assertEqual(result[0].alignment_reason, reason)

    def test_combined_alignment_precedes_contextual_pos(self):
        kb = ["disj(running, children)"]
        premise, hypothesis = "children", "running"
        pos_only = run(kb, premise, hypothesis, config(lemmatization_mode="replacement"))
        combined = run(kb, premise, hypothesis,
                       config(lemmatization_mode="replacement", argument_alignment_mode="replacement"))
        self.assertEqual(pos_only[0].relation, "disj(running, children)")
        self.assertEqual(combined[0].relation, "disj(child, run)")
        self.assertEqual(combined[0].transformations, ["argument_alignment", "lemmatization_pos"])

    def test_final_ph_filter_is_independent(self):
        base = config(argument_alignment_mode="replacement")
        self.assertEqual(len(run(["isa_wn(dog, animal)"], "cat", "bird", base)), 1)
        self.assertEqual(run(["isa_wn(dog, animal)"], "cat", "bird",
                             FilteringConfig(**(base.model_dump() | {"final_ph_filter": True}))), [])

    def test_underscore_and_preposition_modes(self):
        for mode, expected in (
            ("off", ["isa_wn(blue_shirt, clothing)"]),
            ("additive", ["isa_wn(blue_shirt, clothing)", "isa_wn(blue shirt, clothing)"]),
            ("replacement", ["isa_wn(blue shirt, clothing)"]),
        ):
            result = run(["isa_wn(blue_shirt, clothing)"], "", "",
                         config(underscores_mode=mode, leading_preposition_mode="off"))
            self.assertEqual([r.relation for r in result], expected)
        for mode, expected in (
            ("off", ["isa_wn(in blue shirt, clothing)"]),
            ("additive", ["isa_wn(in blue shirt, clothing)", "isa_wn(blue shirt, clothing)"]),
            ("replacement", ["isa_wn(blue shirt, clothing)"]),
        ):
            result = run(["isa_wn(in blue shirt, clothing)"], "", "",
                         config(leading_preposition_mode=mode))
            self.assertEqual([r.relation for r in result], expected)

    def test_historical_verb_additive_dag_provenance_and_order(self):
        settings = FilteringConfig.operational(lemmatization_kind="verb", final_ph_filter=False)
        output = run(["isa_wn(running cats, running dogs)"], "", "", settings)
        self.assertEqual([(r.relation, r.provenance) for r in output], [
            ("isa_wn(running cats, running dogs)", "llm"),
            ("isa_wn(running dogs, running cats)", "derived_swap"),
            ("isa_wn(run cats, run dogs)", "derived_lemma"),
            ("isa_wn(run dogs, run cats)", "derived_lemma_swap"),
            ("isa_wn(cats, dogs)", "derived_diff"),
            ("isa_wn(dogs, cats)", "derived_diff_swap"),
        ])

    def test_replacement_ancestors_do_not_return_via_diff(self):
        output = run(["isa_wn(children running, children smiling)"],
                     "children running", "children smiling",
                     config(lemmatization_mode="replacement", diff_only_mode="additive"))
        self.assertEqual([r.relation for r in output], [
            "isa_wn(child run, child smile)", "isa_wn(run, smile)",
        ])

    def test_diff_replacement_keeps_unchanged_if_not_applicable(self):
        output = run(["isa_wn(red cat, blue cat)", "isa_wn(dog, animal)"], "", "",
                     config(diff_only_mode="replacement"))
        self.assertEqual([r.relation for r in output], ["isa_wn(red, blue)", "isa_wn(dog, animal)"])

    def test_dedup_and_identical_arguments_have_audits(self):
        events = []
        output = run(["isa_wn(dog, animal)"] * 2 + ["isa_wn(dog, dog)", "other(a, b)", "bad"],
                     "", "", config(), events)
        self.assertEqual(len(output), 1)
        self.assertEqual({event["reason"] for event in events},
                         {"duplicate", "identical_arguments", "unsupported_predicate", "invalid_syntax"})

    def test_morphological_match_does_not_change_kb(self):
        kb = ["isa_wn(child, run)"]
        for policy, expected in (("exact", 0), ("exact_or_lemma", 1), ("lemma_only", 1)):
            output = run(kb, "children", "running",
                         config(lemma_match_policy=policy, final_ph_filter=True))
            self.assertEqual(len(output), expected)
            if output:
                self.assertEqual(output[0].relation, kb[0])

    def test_matching_preserves_same_pos_and_all_components_contract(self):
        ctx = SmallContext("blue very large shirt", "")
        self.assertTrue(ctx.matches("blue shirt", ctx.premise, "exact"))
        self.assertFalse(ctx.matches("blue shirt", "blue", "exact"))
        # No noun-to-verb cross matching: saw's noun remains saw, verb is see.
        self.assertFalse(ctx.matches("saw", "see", "exact"))
        self.assertTrue(ctx.matches("saw", "see", "exact_or_lemma"))

    def test_context_reuse_preserves_outputs_and_reuses_indices(self):
        ctx = SmallContext("children", "running")
        settings = config(argument_alignment_mode="replacement")
        first = run(["isa_wn(running, children)"], ctx.premise, ctx.hypothesis, settings, context=ctx)
        misses = ctx.index.cache_info().misses
        second = run(["isa_wn(running, children)"], ctx.premise, ctx.hypothesis, settings, context=ctx)
        self.assertEqual(first, second)
        self.assertEqual(ctx.index.cache_info().misses, misses)

    def test_off_path_loads_no_resources(self):
        settings = config(underscores_mode="off", leading_preposition_mode="off")
        with patch("nltk.download", side_effect=AssertionError("network")), \
             patch("nltk.data.find", side_effect=AssertionError("unnecessary resource")):
            self.assertEqual(prepare_filtering_resources([settings]), {})
            output = pipeline_filter_kb_injections(["isa_wn(dog, animal)"], "", "",
                                                   filtering=settings, offline=True)
        self.assertEqual(output[0].relation, "isa_wn(dog, animal)")

    def test_missing_offline_resources_fail_without_downloading(self):
        with patch("nltk.data.find", side_effect=LookupError("missing")), \
             patch("nltk.download", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(LookupError, "No download was attempted"):
                prepare_filtering_resources([config()])
            with self.assertRaisesRegex(LookupError, "No download was attempted"):
                pipeline_filter_kb_injections(["isa_wn(dog, animal)"], "", "",
                                               filtering=config(), offline=True)

    def test_legacy_alias_conflicts_and_context_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "post_process conflicts"):
            pipeline_filter_kb_injections([], "", "", post_process=False, filtering=config())
        with self.assertRaisesRegex(ValueError, "different premise"):
            pipeline_filter_kb_injections([], "one", "", filtering=config(), context=SmallContext("two", ""))
        with self.assertRaisesRegex(ValueError, "offline=True"):
            pipeline_filter_kb_injections([], "", "", filtering=config(), context=SmallContext("", ""), offline=True)

    def test_legacy_wrapper_warns_about_ignored_swap(self):
        with patch("kbprojection.filtering.filter_candidates", return_value=[]) as mocked:
            with self.assertWarns(DeprecationWarning):
                filter_kb_by_prem_hyp([], "", "", swap_args=True)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
