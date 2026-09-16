import unittest

from calculate_multi_reference_f1 import (
    argument_order_agnostic_relation_counts,
    evaluate_prediction_column,
    parse_kb_cell,
    relation_counts,
)


class ModelAnswerExclusionTests(unittest.TestCase):
    def test_exclusions_empty_entailment_and_final_answer(self):
        rows = [
            {"KB": "NO_RELATION", "raw_response": "answer: entailment\nrelations: { }"},
            {"KB": "NO_RELATION", "raw_response": "answer: non-entailment"},
            {"KB": "isa_wn(dog, animal)", "raw_response": "answer: entailment\nanswer: non-entailment"},
            {"KB": "NO_RELATION", "error": "provider failed"},
            {"KB": ""},
        ]
        for row in rows:
            row["human"] = "NO_RELATION"
        result = evaluate_prediction_column(rows, "KB", ["human"], empty_prediction_is_no_relation=False)
        self.assertEqual(result.evaluated_items, 1)
        self.assertEqual(result.exact_best_matches, 1)
        self.assertEqual(result.skipped_non_entailment, 2)
        self.assertEqual(result.skipped_error, 1)
        self.assertEqual(result.skipped_missing_prediction, 1)

    def test_wide_model_columns_and_human_annotations(self):
        row = {"LLM__lasha__model_KB": "NO_RELATION",
               "LLM__lasha__model_raw_response": "answer: non-entailment",
               "Jorryt_KB": "NO_RELATION", "Lasha_KB": "NO_RELATION",
               "raw_response": "answer: non-entailment"}
        model = evaluate_prediction_column([row], "LLM__lasha__model_KB", ["Lasha_KB"], empty_prediction_is_no_relation=False)
        human = evaluate_prediction_column([row], "Jorryt_KB", ["Lasha_KB"], empty_prediction_is_no_relation=False)
        self.assertEqual(model.skipped_non_entailment, 1)
        self.assertEqual(human.exact_best_matches, 1)

    def test_always_wrong_policy_keeps_non_entailment_in_denominator(self):
        rows = [
            {"KB": "NO_RELATION", "raw_response": "answer: non-entailment", "human": "NO_RELATION"},
            {"KB": "isa_wn(dog, animal)", "raw_response": "answer: non-entailment", "human": "isa_wn(dog, animal)"},
        ]
        result = evaluate_prediction_column(
            rows, "KB", ["human"], empty_prediction_is_no_relation=False,
            non_entailment_policy="always-wrong",
        )
        self.assertEqual(result.evaluated_items, 2)
        self.assertEqual(result.scored_non_entailment_as_wrong, 2)
        self.assertEqual(result.skipped_non_entailment, 0)
        self.assertEqual(result.exact_best_matches, 0)
        self.assertEqual((result.selected_counts.tp, result.selected_counts.fp, result.selected_counts.fn), (0, 2, 1))


class ArgumentOrderAgnosticMetricTests(unittest.TestCase):
    def test_reversed_arguments_only_match_in_diagnostic_metric(self):
        prediction = parse_kb_cell("isa_wn(fruit, apple)")
        reference = parse_kb_cell("isa_wn(apple, fruit)")

        directed = relation_counts(prediction, reference)
        agnostic = argument_order_agnostic_relation_counts(prediction, reference)

        self.assertEqual((directed.tp, directed.fp, directed.fn), (0, 1, 1))
        self.assertEqual((agnostic.tp, agnostic.fp, agnostic.fn), (1, 0, 0))

    def test_unrelated_pairs_remain_distinct(self):
        prediction = parse_kb_cell("isa_wn(apple, vehicle)")
        reference = parse_kb_cell("isa_wn(apple, fruit)")

        agnostic = argument_order_agnostic_relation_counts(prediction, reference)

        self.assertEqual((agnostic.tp, agnostic.fp, agnostic.fn), (0, 1, 1))

    def test_best_reference_is_reselected_for_agnostic_metric(self):
        rows = [
            {
                "ID": "example",
                "prediction": "isa_wn(fruit, apple)",
                "reference_a": "isa_wn(apple, fruit)",
                "reference_b": "isa_wn(car, vehicle)",
            }
        ]

        result = evaluate_prediction_column(
            rows,
            "prediction",
            ["reference_a", "reference_b"],
            empty_prediction_is_no_relation=False,
            calculate_argument_order_agnostic=True,
        )

        self.assertEqual(
            (
                result.selected_counts.tp,
                result.selected_counts.fp,
                result.selected_counts.fn,
            ),
            (0, 1, 1),
        )
        self.assertIsNotNone(result.argument_order_agnostic_counts)
        self.assertEqual(
            (
                result.argument_order_agnostic_counts.tp,
                result.argument_order_agnostic_counts.fp,
                result.argument_order_agnostic_counts.fn,
            ),
            (1, 0, 0),
        )
        self.assertEqual(result.argument_order_agnostic_exact_best_matches, 1)

    def test_agnostic_exact_match_requires_the_complete_relation_set(self):
        rows = [
            {
                "ID": "example",
                "prediction": "isa_wn(fruit, apple)",
                "reference": "isa_wn(apple, fruit); isa_wn(cat, animal)",
            }
        ]

        result = evaluate_prediction_column(
            rows,
            "prediction",
            ["reference"],
            empty_prediction_is_no_relation=False,
            calculate_argument_order_agnostic=True,
        )

        self.assertEqual(result.argument_order_agnostic_exact_best_matches, 0)


if __name__ == "__main__":
    unittest.main()
