# Gemini 3.1 Flash-Lite proof differences

This report accompanies `gemini_lex_proof_differences.csv`. The CSV was
computed from the Gemini 3.1 Flash-Lite `lex` LangPro artifacts for the
1,000-item run.

## Configurations

- `WN`: WordNet only (`wordnet_only.json`)
- `LLM1`: LLM lexical knowledge, one-shot (`llm_only_one_shot.json`)
- `LLMa`: LLM lexical knowledge, agentic (`llm_only_agentic.json`)
- `WN1`: WordNet plus LLM one-shot (`wordnet_llm_one_shot.json`)
- `WNa`: WordNet plus LLM agentic (`wordnet_llm_agentic.json`)

Each comparison column contains `1` when the method on the left solved the
problem and the method on the right did not solve it. Empty cells are coded as
`0`. The CSV therefore records problem-level gains, not the net deltas shown
in the aggregate table.

## Aggregate counts

| Configuration | Solved problems |
|---|---:|
| WN | 157 |
| LLM1 | 165 |
| LLMa | 180 |
| WN1 | 179 |
| WNa | 193 |

The aggregate table deltas are:

```text
LLM1 − WN       = 165 − 157 = +8
LLMa − WN       = 180 − 157 = +23
LLMa − LLM1     = 180 − 165 = +15
WN1 − WN        = 179 − 157 = +22
WNa − WN        = 193 − 157 = +36
WNa − WN1       = 193 − 179 = +14
```

The first two deltas are net values. For example, `LLM1 > WN` has 26 gains
and 18 losses, giving a net increase of 8. Consequently, the `1>WN` CSV
column contains 26 ones rather than 8. The CSV preserves this distinction so
that the underlying problem-level evidence is not hidden.

## Illustrative examples

### Multiple lexical bridges

Problem `10101477.jpg#4r3e`:

- Premise: “Asian man sweeping the walkway.”
- Hypothesis: “A man cleaning the floor.”
- Useful relations: `sweep → clean` and `walkway → floor`.
- WordNet alone fails, while the LLM-assisted configurations solve it.

### A single verb relation

Problem `171211612.jpg#4r3e`:

- Premise: “A man holding a camera underwater.”
- Hypothesis: “A man is using a camera.”
- Useful relation: `hold → use`.
- One missing lexical relation changes the result from `neutral` to
  `entailment`.

### Agentic recovery of a missing relation

Problem `2282600972.jpg#2r1e`:

- Premise: “Two skiers stand near a snowy mountain.”
- Hypothesis: “Some humans standing.”
- One-shot knowledge finds `skier → human` but misses the verb-form relation.
- The agentic method adds `stand → standing` and succeeds.

### Correcting an overly specific relation

Problem `3197194113.jpg#0r2e`:

- Premise: a man is swallowing a sword.
- Hypothesis: “A man is performing.”
- The first attempt proposes `swallow sword → perform`.
- The agentic method refines this to `swallow → perform`, which supports the
  proof.

### Several small relations working together

Problem `2930616480.jpg#0r2e`:

- Premise: “A female is wearing a big purple flower in her hair.”
- Hypothesis: “A girl with a flower in her hair.”
- The agentic method adds `female → girl`, `wear → with`, and `have → with`.
- Together these relations make the WordNet-plus-agentic configuration succeed.

## Coverage

The CSV has 1,000 rows. There are 47 unique problems with at least one
problem-level gain in the six comparison columns. If losses are also included
(problems solved by the right-hand method but not the left-hand method), the
union contains 63 unique changed problems.
