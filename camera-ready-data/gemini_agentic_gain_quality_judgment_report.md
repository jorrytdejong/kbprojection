# Gemini Agentic-Gain Quality Judgment Report

## Purpose

This report reviews the problems newly solved by Gemini 3.1 Flash-Lite after
agentic refinement. It covers both comparisons reported in the paper:

- **+14:** WordNet + LLM, agentic versus one-shot.
- **+15:** LLM only, agentic versus one-shot.

The judgment concerns the semantic quality of the added or changed relations,
not merely whether LangPro succeeded. These judgments are preliminary and
should be confirmed by the authors or annotators.

## Preliminary summary

| Comparison | Clearly good | Mixed | Poor | Total |
|---|---:|---:|---:|---:|
| WordNet + LLM (+14) | 0 | 0 | 14 | 14 |
| LLM only (+15) | 2 | 2 | 11 | 15 |

Here, **poor** includes morphological relations, modifier dropping,
context-dependent relations, prover-specific shortcuts, and semantically
incorrect relations.

## The +14 cases: WordNet + LLM

| Problem ID and added relations | Judgment |
|---|---|
| `2282600972`: `stand ⊑ standing`; `standing ⊑ stand` | Poor — morphological rather than lexical knowledge |
| `254527963`: `black dog ⊑ dog`; `running along ⊑ running on`; `run along ⊑ run on`; `along ⊑ on` | Poor — modifier dropping and context-dependent relations |
| `2740013516`: `standing outside ⊑ outdoors`; `stand outside ⊑ outdoors` | Poor — context-dependent |
| `2930616480`: `wear ⊑ with`; `have ⊑ with` | Poor — prover-specific predicate shortcuts |
| `3197194113`: `swallow ⊑ perform` | Poor — only plausible in this particular context |
| `3522349685`: `little girl ⊑ girl`; `petting ⊑ has`; `pet ⊑ have` | Poor — modifier dropping and context-dependent relations |
| `3706356018`: `little kids ⊑ children`; `little kid ⊑ children`; `amusement park ride ⊑ ride` | Poor — inflection and modifier dropping |
| `3709301369`: `in ⊑ wearing`; `in ⊑ wear` | Poor — context-dependent |
| `4151285133`: `on ⊑ in`; `mirror on the wall ⊑ mirror` | Poor — preposition substitution and modifier dropping |
| `470458461`: `push ⊑ have` | Poor — pushing does not generally entail possession |
| `856985136`: `put into ⊑ dip into`; `put ⊑ dip` | Poor — context-dependent |
| `vg_len95r3e`: `tennis player ⊑ tennis playing`; `tennis player ⊑ tennis play`; `player ⊑ playing` | Poor — category mismatch |
| `vg_verb119r2e`: `adjusting ⊑ fixing`; `adjusting for accuracy ⊑ fixing`; `adjust for accuracy ⊑ fix` | Poor — morphological/context-dependent duplicates |
| `vg_verb51r3e`: `grasp ⊑ rest` | Poor — semantically incorrect |

## The +15 cases: LLM only

| Problem ID and added relations | Judgment |
|---|---|
| `1357753846`: `black dog ⊑ animal`; `stream ⊑ water` | Mixed — `stream ⊑ water` is good, but `black dog ⊑ animal` drops a modifier |
| `2282600972`: `stand ⊑ standing`; `standing ⊑ stand` | Poor — morphological rather than lexical knowledge |
| `254527963`: `black dog ⊑ dog`; `running along ⊑ running on`; `run along ⊑ run on`; `along ⊑ on` | Poor — modifier dropping and context-dependent relations |
| `2549933281`: `woman ⊑ women`; `women ⊑ woman`; `in ⊑ wear`; `floral wedding dress ⊑ wedding dress`; `floral wed dress ⊑ wed dress` | Poor — inflection, modifier dropping, and malformed relations |
| `2688731661`: `lady ⊑ woman` | Good — generally valid lexical entailment |
| `2740013516`: `standing outside ⊑ outdoors`; `stand outside ⊑ outdoors` | Poor — context-dependent |
| `3197194113`: `swallowing ⊑ performing`; `swallow ⊑ perform` | Poor — context-dependent |
| `3419634480`: `run ⊑ outdoors` | Poor — running does not generally entail being outdoors |
| `3522349685`: `petting ⊑ having`; `pet ⊑ have` | Poor — context-dependent predicate shortcuts |
| `3706356018`: `little kids ⊑ children`; `little kid ⊑ children`; `amusement park ride ⊑ ride` | Poor — inflection and modifier dropping |
| `470458461`: `push ⊑ have` | Poor — prover-specific shortcut |
| `4981040278`: `desk ⊑ furniture` | Good refinement — valid relation, although the one-shot relation `desk ⊑ piece of furniture` was already semantically good |
| `86124605`: `lady ⊑ woman`; `making ⊑ cooking`; `make ⊑ cook` | Mixed — `lady ⊑ woman` is good, but the other relations are morphological or context-dependent |
| `vg_verb119r2e`: `adjusting ⊑ fixing`; `adjusting for accuracy ⊑ fixing`; `adjust for accuracy ⊑ fix` | Poor — morphological/context-dependent duplicates |
| `vg_verb51r3e`: `grasp ⊑ rest` | Poor — semantically incorrect |

## Interpretation

The proof-coverage gains should not be described as equivalent gains in
high-quality lexical knowledge. The agentic method often improves LangPro's
ability to complete a proof by adding relations that are useful for the
specific sentence but are not generally valid lexical entailments.

Before using the counts in the paper, the authors should confirm whether
mixed cases are counted as valid, poor, or reported separately.
