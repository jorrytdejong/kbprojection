"""Scientific contracts for symmetric scoring; no NLP downloads or services."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calculate_multi_reference_f1 import evaluate_prediction_column, parse_kb_cell, relation_counts
from kbprojection.scoring import ScoringConfig, ScoringContext
from scripts.experiments.evaluate_kb_scoring import build_parser, configurations, evaluate_rows, run


class FakeNLP:
    """Explicit POS/lemma fixtures; no assumptions about an installed tagger."""
    def __init__(self, *args, **kwargs):
        pass

    def tokens(self, text):
        return text.split()

    def tags(self, text):
        tags = {"dogs": "NNS", "animals": "NNS", "run": "VBP", "dogs_noun": "NN"}
        if text == "They saw":
            return (("They", "PRP"), ("saw", "VBD"))
        if text == "A saw":
            return (("A", "DT"), ("saw", "NN"))
        return tuple((word, tags.get(word, "NN")) for word in text.split())

    def lemma(self, word, pos):
        return {("dogs", "n"): "dog", ("animals", "n"): "animal", ("saw", "v"): "see"}.get((word, pos), word)


class TestSymmetricScoring(unittest.TestCase):
    def test_four_conditions_require_both_lemma_and_direction_relaxation(self):
        context = ScoringContext("dogs run", "animals run")
        context._nlp = FakeNLP()
        prediction, reference = parse_kb_cell("(animal, dog)"), parse_kb_cell("(dogs, animals)")
        for config, expected in [(ScoringConfig(), 0), (ScoringConfig(True), 0),
                                 (ScoringConfig(False, True), 0), (ScoringConfig(True, True), 1)]:
            with self.subTest(config=config):
                left, right = context.kb(prediction, config), context.kb(reference, config)
                self.assertEqual(relation_counts(left, right).tp, expected)
                self.assertEqual(relation_counts(right, left).tp, expected)
        self.assertEqual(reference, frozenset({("dogs", "animals")}))

    def test_context_is_symmetric_and_conservative(self):
        context = ScoringContext("They saw", "A saw")
        context._nlp = FakeNLP()
        self.assertEqual(context.argument("saw"), ("saw", "ambiguous"))
        self.assertEqual(context.argument("unseen"), ("unseen", "absent"))
        for p, h in [("dogs run", "animals run"), ("animals run", "dogs run")]:
            context = ScoringContext(p, h)
            context._nlp = FakeNLP()
            self.assertEqual(context.kb(parse_kb_cell("(dogs, animals)"), ScoringConfig(True)),
                             frozenset({("dog", "animal")}))

    def test_baseline_retains_self_pairs_and_legacy_arity(self):
        context = ScoringContext()
        raw = "isa_wn(a,a); disj(a,b,c); (a, ); (a,a)"
        self.assertEqual(context.kb(parse_kb_cell(raw), ScoringConfig()), parse_kb_cell(raw))
        self.assertIn(("a", "a"), context.kb(parse_kb_cell(raw), ScoringConfig(False, True)))
        self.assertNotEqual(context.kb(parse_kb_cell("(a,b,c)"), ScoringConfig(False, True)),
                            context.kb(parse_kb_cell("(c,b,a)"), ScoringConfig(False, True)))
        self.assertEqual(len(context.kb(parse_kb_cell("(a,b);(b,a)"), ScoringConfig(False, True))), 1)

    @patch("kbprojection.filtering_context.FilteringContext", FakeNLP)
    def test_existing_column_scorer_applies_rule_to_references_and_sequence(self):
        rows = [{"premise": "dogs run", "hypothesis": "animals run", "P_KB": "(animal,dog)", "R_KB": "(dogs,animals)"}]
        before = copy.deepcopy(rows)
        result = evaluate_prediction_column(rows, "P_KB", ["R_KB"], empty_prediction_is_no_relation=False,
                                            scoring=ScoringConfig(True, True), calculate_position_sensitive=True)
        self.assertEqual(result.selected_counts.tp, 1)
        self.assertEqual(result.position_sensitive_counts.tp, 1)
        self.assertEqual(rows, before)

    @patch("kbprojection.filtering_context.FilteringContext", FakeNLP)
    def test_lemma_scoring_feeds_the_upstream_order_diagnostic(self):
        rows = [{"premise": "dogs run", "hypothesis": "animals run",
                 "P_KB": "(animal,dog)", "R_KB": "(dogs,animals)"}]
        result = evaluate_prediction_column(
            rows, "P_KB", ["R_KB"], empty_prediction_is_no_relation=False,
            scoring=ScoringConfig(lemmatize=True),
            calculate_argument_order_agnostic=True,
        )
        self.assertEqual((result.selected_counts.tp, result.selected_counts.fp,
                          result.selected_counts.fn), (0, 1, 1))
        self.assertIsNotNone(result.argument_order_agnostic_counts)
        self.assertEqual((result.argument_order_agnostic_counts.tp,
                          result.argument_order_agnostic_counts.fp,
                          result.argument_order_agnostic_counts.fn), (1, 0, 0))
        self.assertEqual(result.argument_order_agnostic_exact_best_matches, 1)

    @patch("kbprojection.filtering_context.FilteringContext", FakeNLP)
    def test_replay_keeps_raw_kbs_and_excludes_same_slots(self):
        sources = [{"ID": "1", "premise": "dogs run", "hypothesis": "animals run", "R_KB": "(dogs,animals)"},
                   {"ID": "2", "premise": "", "hypothesis": "", "R_KB": "NO_RELATION"}]
        saved = [{**sources[0], "model": "m", "prompt": "p", "repeat": "1", "KB": "(animal,dog)", "error": ""},
                 {**sources[1], "model": "m", "prompt": "p", "repeat": "1", "KB": "NO_RELATION", "error": ""},
                 {**sources[0], "model": "m", "prompt": "p", "repeat": "2", "KB": "(x,y)", "error": "failure"}]
        before = copy.deepcopy((saved, sources))
        configs = configurations(build_parser().parse_args(["--compare"]))
        details, metrics, summary, examples, statuses, decisions = evaluate_rows(saved, sources, configs, ["R_KB"], 2)
        self.assertEqual(statuses, {"valid": 2, "error": 1, "missing": 1})
        self.assertEqual(list(decisions), sorted(decisions))
        self.assertEqual((saved, sources), before)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(summary), 4)
        combined = next(r for r in details if r["ID"] == "1" and r["repeat"] == 1 and r["configuration"] == "combined")
        self.assertEqual((combined["tp"], combined["fp"], combined["fn"]), (1, 0, 0))
        self.assertEqual(combined["raw_reference_KB"], "(dogs,animals)")
        self.assertTrue(any(r["changed_side"] == "reference" for r in examples))
        for row in metrics:
            self.assertEqual(row["evaluated_items"], 2 if row["repeat"] == 1 else 0)

    def test_print_export_and_flag_contract(self):
        with self.assertRaisesRegex(ValueError, "individual"):
            configurations(build_parser().parse_args(["--compare", "--lemmatize"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "refs.csv").write_text("ID,premise,hypothesis,R_KB\n1,P,H,\"(a,a)\"\n", encoding="utf-8")
            (root / "saved.csv").write_text("ID,premise,hypothesis,model,prompt,repeat,KB,error\n1,P,H,m,p,1,\"(a,a)\",\n", encoding="utf-8")
            args = build_parser().parse_args(["--input-csv", str(root / "saved.csv"), "--sample-csv", str(root / "refs.csv"), "--reference-columns", "R_KB", "--repeats", "1"])
            repository = Path(__file__).resolve().parents[1]
            # Baseline exports must also work without site-packages/NLP libraries.
            subprocess.run([sys.executable, "-S", str(repository / "scripts/experiments/evaluate_kb_scoring.py"),
                            "--input-csv", str(root / "saved.csv"), "--sample-csv", str(root / "refs.csv"),
                            "--reference-columns", "R_KB", "--repeats", "1", "--output", str(root / "stdlib")],
                           check=True, capture_output=True, text=True)
            with patch("sys.stdout", new_callable=io.StringIO), patch("nltk.download", side_effect=AssertionError("download")):
                run(args)
                args.output = root / "out"
                run(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["nlp_resources"], {})
            self.assertFalse(manifest["stored_annotations_modified"])
            for name in manifest["outputs_sha256"]:
                self.assertNotIn(b"\r\n", (args.output / name).read_bytes())
            with self.assertRaisesRegex(ValueError, "fresh"):
                run(args)
