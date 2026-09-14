# Gemini 3.1 Flash-Lite qualitative contrast examples

These examples illustrate wrong-to-correct cases from the saved SNLI-1K results. The contrast JSON files contain the full selected sets, predictions, and available KB traces.

## WordNet only vs. LLM only

**Problem `10101477.jpg#4r3e`**

- Premise: “Asian man sweeping the walkway.”
- Hypothesis: “A man cleaning the floor.”
- WordNet only: `neutral` (incorrect)
- LLM only: `entailment` (correct)
- LLM KB: `isa_wn(sweep, clean)`; `isa_wn(walkway, floor)`

The generated KB supplies links for both the action and the surface, which helps connect the premise to the hypothesis. See [the WordNet-only vs. LLM-only JSON](flash_lite_lex_wordnet_only_vs_llm_only_one_shot.json).

## WordNet only vs. WordNet + LLM, one-shot

**Problem `1207159468.jpg#3r1e`**

- Premise: “A boy and a person suspended above a dome structure.”
- Hypothesis: “The boy is above a building.”
- WordNet only: `neutral` (incorrect)
- WordNet + LLM, one-shot: `entailment` (correct)
- Recorded generated KB: `isa_wn(dome structure, building)`; `isa_wn(dome structure, build)`

Here the LLM-generated relation bridges “dome structure” and “building.” See [the WordNet-only vs. WordNet + LLM JSON](flash_lite_lex_wordnet_only_vs_wordnet_llm_one_shot.json).

## LLM-only one-shot vs. LLM-only agentic

**Problem `1357753846.jpg#1r1e`**

- Premise: “A black dog is walking through a stream of water.”
- Hypothesis: “An animal is walking through water.”
- LLM-only one-shot: `neutral` (incorrect), with `isa_wn(dog, animal)`
- LLM-only agentic: `entailment` (correct), after revising the KB to `isa_wn(black dog, animal)` and `isa_wn(stream, water)`

The retry changes the KB and the prediction. See [the Stefan LLM-only one-shot vs. agentic JSON](flash_lite_stefan_llm_only_one_shot_vs_llm_only_agentic.json); the same problem also appears in the lex comparison file.

## WordNet + LLM one-shot vs. WordNet + LLM agentic

**Problem `195065912.jpg#3r2e`**

- Premise: “A teenage boy performs a stunt on his skateboard in a skate park.”
- Hypothesis: “The boy is a skateboarder.”
- WordNet + LLM, one-shot: `neutral` (incorrect), with `isa_wn(skateboard, skateboarder)`
- WordNet + LLM, agentic: `entailment` (correct), after revising the relation to `isa_wn(teenage boy, skateboarder)`

The retry changes the relation’s subject from the object “skateboard” to the person “teenage boy.” See [the Stefan one-shot vs. agentic JSON](flash_lite_stefan_wordnet_llm_one_shot_vs_wordnet_llm_agentic.json); the lex file contains the same case as well.

## LLM-only one-shot vs. WordNet + LLM one-shot

**Problem `1357753846.jpg#1r1e`**

- Premise: “A black dog is walking through a stream of water.”
- Hypothesis: “An animal is walking through water.”
- LLM-only one-shot: `neutral` (incorrect)
- WordNet + LLM one-shot: `entailment` (correct)

For this record, the WordNet + LLM run was already correct at its WordNet baseline, so it skipped KB generation. Its empty generated-KB trace does not mean the run had no WordNet knowledge. See [the lex one-shot comparison JSON](flash_lite_lex_llm_only_one_shot_vs_wordnet_llm_one_shot.json).

## Interpretation note

The WordNet-only result files record predictions, but do not preserve the per-problem WordNet KB. For WordNet-only contrasts, the JSON therefore marks that KB as unavailable rather than reconstructing or guessing its contents. The one-shot vs. agentic traces do preserve the KB submitted on each attempt.
