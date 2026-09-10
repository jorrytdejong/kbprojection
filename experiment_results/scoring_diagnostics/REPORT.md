# KB scoring diagnostics

Saved predictions and annotations are not rewritten. Only their scoring representations change. No filtering, P/H reorientation, underscore replacement, preposition removal or identical-argument suppression is applied.

Slots: 14,480; status counts: {'error': 38, 'valid': 14442}.

Tables report percentages as mean (± sample SD) across generation runs, rounded to one decimal place. Rows are sorted by descending mean micro-F1 before rounding; bold means mark the highest unrounded value in each column. Five generation runs are not five-shot prompting.

Micro-precision, micro-recall and micro-F1 use TP, FP and FN aggregated across evaluated problems within each run. Exact match is the percentage of evaluated problems whose entire scored KB equals at least one reference; partial matches do not count.

Only the eight evaluated LLMs are included. WordNet and other baseline systems from the paper are not part of this replay.

## Original

Prompt: `lasha`.

| Model | Exact match | Micro-F1 ↓ | Micro-Precision | Micro-Recall |
|---|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite | 65.4 (± 0.5) | **76.2** (± 0.6) | **70.7** (± 0.5) | 82.6 (± 0.6) |
| Claude Sonnet 4.5 | 64.0 (± 0.2) | 75.9 (± 0.2) | 69.1 (± 0.2) | **84.2** (± 0.3) |
| GPT-5.4 | **66.1** (± 0.9) | 74.1 (± 0.3) | 68.8 (± 0.4) | 80.2 (± 0.7) |
| Gemini 3.5 Flash | 60.9 (± 0.5) | 73.4 (± 0.5) | 66.0 (± 0.5) | 82.6 (± 0.5) |
| Claude Haiku 4.5 | 57.2 (± 0.0) | 70.6 (± 0.0) | 62.1 (± 0.0) | 81.8 (± 0.0) |
| GPT-OSS-20B | 54.2 (± 1.1) | 61.5 (± 0.9) | 60.1 (± 0.8) | 63.0 (± 1.3) |
| GPT-5.4 Mini | 46.1 (± 1.1) | 59.3 (± 0.9) | 50.8 (± 0.7) | 71.0 (± 1.5) |
| Gemma 3 4B | 18.0 (± 0.3) | 31.2 (± 0.3) | 21.9 (± 0.2) | 53.8 (± 0.3) |

## Lemmas (both sides)

Prompt: `lasha`.

| Model | Exact match | Micro-F1 ↓ | Micro-Precision | Micro-Recall |
|---|---:|---:|---:|---:|
| Claude Sonnet 4.5 | 65.1 (± 0.2) | **76.9** (± 0.2) | 70.0 (± 0.2) | **85.2** (± 0.3) |
| Gemini 3.1 Flash-Lite | 66.1 (± 0.5) | 76.6 (± 0.6) | **71.4** (± 0.6) | 82.7 (± 0.6) |
| GPT-5.4 | **67.0** (± 1.0) | 74.8 (± 0.3) | 69.6 (± 0.6) | 80.8 (± 0.6) |
| Gemini 3.5 Flash | 61.7 (± 0.5) | 74.4 (± 0.5) | 67.2 (± 0.5) | 83.5 (± 0.5) |
| Claude Haiku 4.5 | 59.1 (± 0.0) | 72.2 (± 0.0) | 63.8 (± 0.0) | 83.1 (± 0.0) |
| GPT-5.4 Mini | 50.9 (± 0.9) | 67.0 (± 0.9) | 57.8 (± 0.9) | 79.7 (± 1.3) |
| GPT-OSS-20B | 58.0 (± 1.1) | 66.6 (± 0.7) | 65.3 (± 0.6) | 67.9 (± 1.3) |
| Gemma 3 4B | 25.4 (± 0.4) | 42.9 (± 0.4) | 30.4 (± 0.4) | 72.5 (± 0.5) |

## Argument order agnostic

Prompt: `lasha`.

| Model | Exact match | Micro-F1 ↓ | Micro-Precision | Micro-Recall |
|---|---:|---:|---:|---:|
| Claude Sonnet 4.5 | 64.4 (± 0.2) | **76.5** (± 0.2) | 69.7 (± 0.1) | **84.8** (± 0.3) |
| Gemini 3.1 Flash-Lite | 65.4 (± 0.5) | 76.2 (± 0.6) | **70.7** (± 0.5) | 82.6 (± 0.6) |
| GPT-5.4 | **66.4** (± 0.9) | 74.5 (± 0.3) | 69.2 (± 0.3) | 80.5 (± 0.7) |
| Gemini 3.5 Flash | 61.2 (± 0.5) | 73.8 (± 0.5) | 66.5 (± 0.5) | 82.9 (± 0.5) |
| Claude Haiku 4.5 | 57.2 (± 0.0) | 70.6 (± 0.0) | 62.1 (± 0.0) | 81.8 (± 0.0) |
| GPT-OSS-20B | 54.2 (± 1.1) | 61.7 (± 0.9) | 60.3 (± 0.8) | 63.2 (± 1.3) |
| GPT-5.4 Mini | 46.2 (± 1.1) | 59.5 (± 0.8) | 51.1 (± 0.7) | 71.3 (± 1.4) |
| Gemma 3 4B | 18.3 (± 0.3) | 31.3 (± 0.3) | 22.0 (± 0.2) | 54.0 (± 0.3) |

## Lemmas + argument order agnostic

Prompt: `lasha`.

| Model | Exact match | Micro-F1 ↓ | Micro-Precision | Micro-Recall |
|---|---:|---:|---:|---:|
| Claude Sonnet 4.5 | 65.5 (± 0.2) | **77.4** (± 0.2) | 70.7 (± 0.1) | **85.7** (± 0.3) |
| Gemini 3.1 Flash-Lite | 66.1 (± 0.5) | 76.6 (± 0.6) | **71.4** (± 0.6) | 82.7 (± 0.6) |
| GPT-5.4 | **67.2** (± 1.0) | 75.2 (± 0.3) | 70.1 (± 0.5) | 81.2 (± 0.6) |
| Gemini 3.5 Flash | 62.0 (± 0.5) | 74.8 (± 0.5) | 67.6 (± 0.5) | 83.8 (± 0.5) |
| Claude Haiku 4.5 | 59.1 (± 0.0) | 72.2 (± 0.0) | 63.8 (± 0.0) | 83.1 (± 0.0) |
| GPT-5.4 Mini | 51.0 (± 1.0) | 67.4 (± 0.9) | 58.2 (± 1.0) | 80.2 (± 1.2) |
| GPT-OSS-20B | 58.0 (± 1.1) | 66.7 (± 0.8) | 65.5 (± 0.7) | 68.1 (± 1.4) |
| Gemma 3 4B | 25.6 (± 0.4) | 43.5 (± 0.4) | 30.9 (± 0.4) | 73.3 (± 0.5) |


## Evaluation contract

Original uses the original directed relation-set parser: case, surrounding whitespace and question marks are normalized; predicate names, duplicates and relation-list order are ignored. Extra-comma tuples and empty arguments retain legacy acceptance. No new syntax rejection is added.

Lemma scoring applies the SAME contextual POS rule to every argument of predictions and every reference. Exact contiguous spans are searched in both premise and hypothesis, independent of argument position. All occurrences must agree on the WordNet POS sequence (noun/verb/adjective/adverb); otherwise the original argument is retained. Missing spans also remain unchanged. POS comes only from P/H, never from a gold relation choice. This is conservative contextual lemma comparison, not exhaustive equivalence of all inflections or proof of semantic equivalence.

Lemma decisions for unique (problem, argument) values across valid predictions and all references: {'absent': 1161, 'ambiguous': 32, 'changed': 636, 'empty': 1, 'unchanged': 1313}.

Argument-order-agnostic scoring sorts the two arguments of binary tuples on BOTH sides. Other tuple arities remain directed. In the combined condition, lemmatization precedes sorting. Canonicalization may merge duplicates, changing denominators; self-pairs remain present.

Each condition selects its own best reference by item F1, TP, -FP, -FN, then reference-column name (maximum). P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Zero denominators are undefined. Empty/empty has item selection score 1 and is an exact match, but adds no relation counts. Missing/error slots are excluded identically. Recall is selected-reference recall, not fixed-reference recall.

These are KB annotation scores, not LangPro proof-success measurements. Changes diagnose sensitivity to scoring conventions; neither a gain nor semantic correctness is guaranteed.

The six deterministic artifacts use UTF-8/LF. The manifest hashes exact output/resource bytes and LF-normalized source/input bytes; local manifests may also include Git metadata. Machine paths, timestamps and environment metadata may differ.
