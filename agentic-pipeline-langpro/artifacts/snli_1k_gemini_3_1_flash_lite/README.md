# Gemini 3.1 Flash-Lite on 1,000 SNLI entailments

This directory preserves the complete, problem-level artifacts for the
Gemini 3.1 Flash-Lite experiment on the fixed 1,000-item SNLI entailment
sample.  The model identifier is `google/gemini-3.1-flash-lite`.

## Canonical saved result set

The following three JSON files are the source of the saved summary in
`summary.xlsx`.  Every file contains 1,000 records and can be used to inspect
individual premise/hypothesis pairs, generated KB relations, LangPro outputs,
and (where applicable) the refinement trace.

| Configuration | One-shot solved | Agentic solved | Artifact |
| --- | ---: | ---: | --- |
| WordNet only | 162 | n/a | `wordnet_only.json` |
| LLM only | 124 | 149 | `llm_only.json` |
| WordNet + LLM | 192 | 202 | `wordnet_plus_llm.json` |

`summary.xlsx` is the compact workbook version of these counts.

## Later LEX rerun

`later_lex_rerun_wordnet_plus_llm.jsonl` is a separate, later complete LEX
rerun of the WordNet-plus-LLM configuration.  It has 1,000 JSONL records and
should not be mixed with the canonical table above: its recorded baseline,
one-shot, and agentic solved counts are 157, 185, and 203, respectively.
It is retained for problem-level analysis and reproducibility.

## File formats

The canonical JSON files use a `records` array.  In LLM runs, each record
contains an `outcome` and a `timeline`; `langpro_with_kb` entries with
`attempt: 1` are the one-shot result and the final `outcome` is the agentic
result.  The rerun is JSON Lines, with one equivalent record per line.
