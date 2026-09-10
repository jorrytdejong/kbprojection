"""Shared offline-replay utility contracts."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calculate_multi_reference_f1 import evaluate_prediction_column, parse_kb_cell
from scripts.experiments.replay_utils import (
    choose_reference, git_source_provenance, prepare_slots, source_sha256,
    summarize,
)


class TestScorerCompatibility(unittest.TestCase):
    def test_selection_matches_existing_scorer_for_legacy_inputs(self):
        cases = [
            ("isa_wn(a, b); disj(c, d); isa_wn(a, b)", "disj(c, d); disj(a, b)", "isa_wn(b, a)"),
            ("isa_wn(b, a)", "isa_wn(a, b)", "(c, d)"),
            ("NO_RELATION", "NO_RELATION", "(a, b)"),
            ("isa_wn(a, b, c)", "(a, b, c)", "(a, b)"),
            ("isa_wn(a, )", "(a, )", "(a, b)"),
            ("(a, b)", "(x, y)", "(z, q)"),
        ]
        for prediction, first, second in cases:
            with self.subTest(prediction=prediction):
                row = {"prediction": prediction, "A_KB": first, "Z_KB": second}
                result = evaluate_prediction_column(
                    [row], "prediction", ["A_KB", "Z_KB"], empty_prediction_is_no_relation=False,
                )
                _, _, reference, counts = choose_reference(
                    parse_kb_cell(prediction),
                    [(column, parse_kb_cell(row[column])) for column in ("A_KB", "Z_KB")],
                )
                self.assertEqual(
                    (counts.tp, counts.fp, counts.fn),
                    (result.selected_counts.tp, result.selected_counts.fp, result.selected_counts.fn),
                )
                self.assertEqual(int(reference == parse_kb_cell(prediction)), result.exact_best_matches)

        _, column, _, _ = choose_reference(
            parse_kb_cell("(a,b)"),
            [("A_KB", parse_kb_cell("(x,y)")), ("Z_KB", parse_kb_cell("(z,q)"))],
        )
        self.assertEqual(column, "Z_KB")
        self.assertNotEqual(parse_kb_cell("(a,b)"), parse_kb_cell("(b,a)"))

    def test_summary_averages_runs_instead_of_pooling(self):
        rows = []
        for run, tp, fp in [(1, 1, 0), (2, 0, 99)]:
            rows.append({
                "prompt": "p", "model": "m", "configuration": "c", "repeat": run,
                "precision": float(tp > 0), "recall": float(tp > 0), "micro_f1": float(tp > 0),
                "exact_best_match_rate": float(tp > 0), "total_items": 1, "evaluated_items": 1,
                "error_runs": 0, "skipped_missing_prediction": 0, "skipped_no_reference": 0,
            })
        result = summarize(rows)[0]
        self.assertEqual(result["mean_micro_f1"], .5)
        self.assertAlmostEqual(result["sample_stddev_micro_f1"], math.sqrt(.5))


class TestReplayUtilities(unittest.TestCase):
    def test_prepare_slots_rejects_duplicate_or_changed_context(self):
        source = {"ID": "1", "premise": "A dog runs.", "hypothesis": "An animal moves."}
        row = {**source, "prompt": "test", "model": "saved-model", "repeat": "1", "KB": "NO_RELATION"}
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            prepare_slots([row, row], [source], 1)
        with self.assertRaisesRegex(ValueError, "premise mismatch"):
            prepare_slots([{**row, "premise": "different"}], [source], 1)

    def test_source_hashes_normalize_lf_and_commit_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.py"
            path.write_bytes(b"first\r\nsecond\n")
            mixed = source_sha256(path)
            path.write_bytes(b"first\nsecond\n")
            self.assertEqual(mixed, source_sha256(path))
        with patch("scripts.experiments.replay_utils.subprocess.check_output", side_effect=[b"abc123\n", b"committed\n"]):
            provenance = git_source_provenance({"example.py": "different"})
        self.assertEqual(provenance["commit"], "abc123")
        self.assertFalse(provenance["code_matches_commit"])
        self.assertEqual(provenance["different_or_untracked_code_paths"], ["example.py"])


if __name__ == "__main__":
    unittest.main()
