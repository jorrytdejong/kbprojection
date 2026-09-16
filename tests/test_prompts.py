import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from kbprojection.prompts import fill_prompt, get_prompt


class TestPrompts(unittest.TestCase):
    def test_icl_prompt_emphasizes_langpro_helpfulness(self):
        prompt = get_prompt("icl")

        self.assertIn("genuinely KB-helpful for LangPro", prompt)
        self.assertIn("Only output relations that are genuinely KB-helpful for LangPro", prompt)
        self.assertIn("Would each fact give LangPro a concrete new bridge", prompt)
        self.assertIn("isa_wn(walk, move around)", prompt)
        self.assertIn("disj(open, closed)", prompt)

    def test_fill_prompt_substitutes_multi_premise_input(self):
        prompt = fill_prompt("cot", ["Premise one.", "Premise two."], "Hypothesis.")

        self.assertIn("Premise: Premise one.\nPremise two.", prompt)
        self.assertIn("Hypothesis: Hypothesis.", prompt)
        self.assertNotIn("${premise}", prompt)
        self.assertNotIn("${hypothesis}", prompt)

    def test_lasha_prompt_is_registered_and_substituted(self):
        prompt = fill_prompt("lasha", ["A dog is running."], "An animal is moving.")

        self.assertIn("answer: entailment", prompt)
        self.assertIn("relations: {", prompt)
        self.assertIn("premise: A dog is running.", prompt)
        self.assertIn("hypothesis: An animal is moving.", prompt)
        self.assertNotIn("${PREMISE}", prompt)
        self.assertNotIn("${HYPOTHESIS}", prompt)

    def test_stefan_prompt_is_registered_and_substituted(self):
        prompt = fill_prompt("stefan", ["A dog is running."], "An animal is moving.")

        self.assertIn("expert in linguistic semantics and logic", prompt)
        self.assertIn("[KB_START]", prompt)
        self.assertIn("premise: A dog is running.", prompt)
        self.assertIn("hypothesis: An animal is moving.", prompt)
        self.assertNotIn("${premise}", prompt)
        self.assertNotIn("${hypothesis}", prompt)


if __name__ == "__main__":
    unittest.main()
