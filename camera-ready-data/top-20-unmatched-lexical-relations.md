# Top 20 unmatched lexical relations from the five strongest models

## Purpose

This analysis makes the lexical-explanation results concrete by identifying
relations that the strongest models frequently generated but that did not occur
in any available human or adjudicated reference for the corresponding NLI
problem.

## Data and filtering

The source was the five-run Lasha output file
`experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_outputs.csv`.
It contains 14,480 prediction slots: 362 problems, eight models, and five
generation runs per model.

The analysis retained only gold-entailment problems and excluded the three
lower-performing models specified for this qualitative analysis:

- Gemma 3 4B;
- GPT-OSS-20B; and
- GPT-5.4 Mini.

This leaves 9,000 prediction slots (360 entailment problems x five models x
five runs). The retained models are Gemini 3.1 Flash-Lite, Claude Sonnet 4.5,
GPT-5.4, Gemini 3.5 Flash, and Claude Haiku 4.5.

For every prediction, each relation was compared against the union of the
available reference LEX sets for that same problem: `Alternative_KB`,
`Ettore_KB`, `Jorryt_KB`, `Lasha_KB`, and `Stefan_KB`. A relation is *unmatched*
when it is absent from all of those reference variants. Relation strings below
are normalized to the `(source, target)` form used by the scorer; direction is
preserved.

## Overall result

The retained predictions contained 10,649 relation occurrences. Of these,
3,462 (32.5%) were unmatched under the definition above. This is not an error
label: an unmatched relation may be a semantically defensible relation that was
unnecessary for a minimal explanation or simply absent from the finite set of
annotated variants.

## Twenty most frequent unmatched relations

| Rank | Relation | Occurrences | Distinct problems |
| ---: | --- | ---: | ---: |
| 1 | `(woman, person)` | 42 | 3 |
| 2 | `(body of water, water)` | 35 | 2 |
| 3 | `(look out over, view)` | 25 | 1 |
| 4 | `(sand beside ocean, beach)` | 25 | 1 |
| 5 | `(kid, human)` | 25 | 1 |
| 6 | `(play instrument, play music)` | 25 | 1 |
| 7 | `(thing, load)` | 25 | 1 |
| 8 | `(highly decorated, nice)` | 25 | 1 |
| 9 | `(tall and green grass, field)` | 25 | 1 |
| 10 | `(man, patient)` | 25 | 1 |
| 11 | `(kiss, show affection)` | 25 | 1 |
| 12 | `(barbie doll, doll)` | 25 | 1 |
| 13 | `(play cello, make music)` | 25 | 1 |
| 14 | `(grassy hillside, outdoors)` | 25 | 1 |
| 15 | `(stuffed duck, duck)` | 25 | 1 |
| 16 | `(man, person)` | 24 | 2 |
| 17 | `(slide off, lose control)` | 24 | 1 |
| 18 | `(massage chair, chair)` | 23 | 1 |
| 19 | `(furry toy, object)` | 23 | 1 |
| 20 | `(jam out, play music)` | 21 | 1 |

Counts near 25 show substantial agreement: there are 25 possible
model-by-run outputs for one problem, so a count of 25 means every retained
model generated that relation in every run for that item. The list therefore
shows a recurring tendency to add plausible hypernyms and broader paraphrases,
such as `(woman, person)`, `(barbie doll, doll)`, and `(kiss, show affection)`,
even when these relations were not included in the reference LEX sets.

## Interpretation boundary

This table should be read as evidence of reference-mismatched or potentially
over-complete generations, not as a manual validity judgment. Qualitative
inspection of the underlying premise, hypothesis, and references is required
before calling any individual relation incorrect.
