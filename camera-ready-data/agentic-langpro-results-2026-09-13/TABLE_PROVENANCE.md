# Expanded LangPro tables: source and version guide

## Scope and metric

The original five configurations become columns, with one row per model and initial prompt. Results are not pooled across populations or export versions. The raw exports were read on 13 September 2026; archive timestamps describe snapshots, not necessarily experiment execution dates.

- **1,000 SNLI problems:** all gold labels are entailment. Count successful entailment predictions/proofs.
- **365 curated problems:** 363 entailments, one contradiction and one neutral. Count correct gold-label predictions. The baseline's 66 successes include the neutral example; these are not 66 entailment proofs.
- **223 human-agreed gold LEX problems:** retain the separate 48/223 to 83/223 oracle table. Direct gold-relation injection uses no LLM prompt. Its JSON and `run_oracle_baseline.py` remain adjacent here.

One-shot means the initial KB attempt with baseline success retained where applicable. Agentic permits at most one refinement; the difference column is agentic minus one-shot. Technical failures and unknown outcomes stay in the fixed denominator. Do not use micro-F1 as proof coverage.

## 1,000-problem counts

| Model | Prompt | WN | LLM one-shot | LLM agentic | Gain | WN+LLM one-shot | WN+LLM agentic | Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite | LEX | 162 | 124 | 149 | +25 | 192 | 202 | +10 |
| Gemini 3.1 Flash-Lite (shared protocol) | LEX | 157 | 165 | 180 | +15 | 179 | 193 | +14 |
| Gemini 3.1 Flash-Lite (shared protocol) | Stefan | 157 | 166 | 181 | +15 | 181 | 197 | +16 |
| GPT-OSS-20B | LEX | 157 | 158 | 194 | +36 | 174 | 212 | +38 |
| GPT-OSS-20B | Stefan | 157 | 168 | 191 | +23 | 187 | 212 | +25 |
| Gemma 3 4B | LEX | 157 | 163 | 163 | 0 | 182 | 182 | 0 |
| Gemma 3 4B | Stefan | 157 | 122 | 122 | 0 | 157 | 157 | 0 |
| GPT-5.4 Mini | LEX | 157 | 162 | 183 | +21 | 178 | 199 | +21 |
| GPT-5.4 Mini | Stefan | 157 | 176 | 188 | +12 | 188 | 203 | +15 |
| Claude Haiku 4.5 | LEX | 157 | 165 | 176 | +11 | 179 | 190 | +11 |
| Claude Haiku 4.5 | Stefan | 157 | 172 | 180 | +8 | 184 | 198 | +14 |

Gemini 3.1 Flash-Lite is **Jorryt's LEX-prompt agentic pipeline**, not Stefan's historical data and not a third prompt arm. The row uses `llm-only-second-run.json`. The earlier export has **125 → 149 (+24)** rather than **124 → 149 (+25)**; only first-attempt success for `7589467042.jpg#0r4e` differs. Both exports give the same final solved-ID set. They are not averaged or treated as independent replications.

All 1,000-problem files have the same 1,000 unique IDs. Baselines differ by version: five predictions marked entailment in Jorryt's WordNet export are unknown in the recent exports, accounting exactly for **162 versus 157**. The recent protocol explicitly changes resource-limit and conflicting-proof interpretation. This is evidence of a version/comparability issue, not evidence that Gemini used a different prompt family. The exact causal contribution of each code change has not been established by a controlled rerun.

## Separate 365-problem counts

| Model | Prompt | WN | LLM one-shot | LLM agentic | Gain | WN+LLM one-shot | WN+LLM agentic | Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash | LEX | 66 | 85 | 121 | +36 | 102 | 134 | +32 |
| Gemini 3.5 Flash | Stefan | 66 | 111 | 125 | +14 | 125 | 136 | +11 |
| GPT-5.4 | LEX | 66 | 88 | 111 | +23 | 106 | 129 | +23 |
| GPT-5.4 | Stefan | 66 | 111 | 118 | +7 | 125 | 133 | +8 |
| Claude Sonnet 4.5 | LEX | 66 | 92 | 116 | +24 | 109 | 128 | +19 |
| Claude Sonnet 4.5 | Stefan | 66 | 107 | 119 | +12 | 123 | 134 | +11 |

These are counts of correct labels on the curated population, not directly comparable with the 1,000-problem totals. The supplied `repair_manifest.json` documents six local-CCG parse repairs and recovered worker failures; it supplements the original protocol's remote-only statement. The input labels are retained, not silently relabelled as entailments.

## Initial prompt and execution provenance

| Rows | Initial prompt identity | Version evidence |
|---|---|---|
| Recent LEX rows | Lasha/LEX, `production_lasha_calibrated` | `protocol.json`: `snapshots/prompts/lasha_calibrated.txt`; SHA-256 `0bd5d7670f2c326db3b242bce6f283a352eba222a86cba06a30c67cf107d689a` |
| Recent Stefan rows | Stefan's knowledge-generation prompt | `protocol.json`: `snapshots/stefan/prompts/knowledge_generation.txt`; no separate initial-template hash recorded in the phase-hash map |
| Jorryt Gemini 3.1 Flash-Lite | LEX, experimenter-confirmed | JSON run `v2` and `with_baseline_v2`, `max_iterations: 2`; exact template revision absent |
| WordNet-only / gold oracle | No LLM prompt | Baseline configuration / gold injection runner |

The recent shared protocol is `extrinsic1000-lex-stefan-pilot-20260911-v1`, source repository revision `1be943f2d6b59693ea2415cbfd33692029e9f8ad`. It uses EasyCCG, RAL 200, WordNet all senses, shared filtering, and the same Stefan critic/refinement templates across the LEX and Stefan initial arms.

The shared critic is `snapshots/prompts/stefan_failure_analysis.txt` (SHA-256 `e8104cf2a830bf76f904e999183937f94e933649a442e6c2b23dc9b3e27d3b59`); refinement is `snapshots/prompts/stefan_knowledge_refinement.txt` (SHA-256 `18c5324375b66e30eeeafb27ea5f0c6ac94aa0b17fc1be5699ac7eed7aac6061`). The hash for `snapshots/prompts/stefan_knowledge_generation.txt` identifies the retry-rules source, not the differently located Stefan initial template.

These snapshot paths identify files in the source archive; the actual prompt texts are **not bundled here**. Exact template identity across the Gemini and recent LEX exports cannot be verified from the shared label alone. Original protocol documents are preserved as received: their wording about historical Table 4 exports does not supersede Jorryt's corrected ownership.

## Where the problem-level predictions are

- Recent 1,000-problem results: `current-1000/<model-slug>/<lex-or-stefan>/{wordnet_only,llm_only_one_shot,llm_only_agentic,wordnet_llm_one_shot,wordnet_llm_agentic}.json`.
- Curated 365-problem results: the same structure under `current-365/`.
- Jorryt Gemini: `gemini-3.1-flash-lite-jorryt-new-prompt-1000/`, with `wordnet-only.json`, both `llm-only-*-run.json` exports, and `llm-and-wordnet.json`.
- Gold oracle: `gold_lex_oracle_223.json`, next to `run_oracle_baseline.py`.
- Each audited row has exact relative source filenames in `proof_coverage_audit.json`.

Agent files contain `problem_id`, `problem`, `outcome`, and `timeline`. Current WordNet-only records instead use flat `id`, `gold_label`, `pred`, and `outcome`. Full raw prover responses are outside this portable archive; trace/source references are not a claim that those raw files are included.

The two `current-1000/google__gemini-3.1-flash-lite/<prompt>/` folders contain the completed shared-protocol Flash Lite experiment: 1,000 problems, both LEX and Stefan, and all five configurations. These results are separate from Jorryt's version above and use the same 157-success WordNet baseline as the other current models.

## Technical failures and limits on interpretation

All 80 recent configuration files used in the table have complete, unique populations; all counts agree with `final_metrics.json`. No current LangPro errors are reported. Some LLM-error counts are substantial:

| Model / prompt | LLM one-shot | LLM agentic | WN+LLM one-shot | WN+LLM agentic |
|---|---:|---:|---:|---:|
| GPT-OSS-20B / LEX | 7 | 75 | 7 | 79 |
| GPT-OSS-20B / Stefan | 57 | 120 | 53 | 128 |
| Gemma 3 4B / LEX | 0 | 638 | 0 | 591 |
| Gemma 3 4B / Stefan | 673 | 679 | 610 | 616 |

The other recent model/prompt configurations report zero LLM errors. In particular, Gemma's zero net refinement gain is not evidence, by itself, that successful refinement requests cannot help. Detailed unknown/error counts remain in `final_tables.txt` and the audit JSON.

## Original paper numbers

The original row is WN **162**, LLM **97 → 195 (+98)**, WN+LLM **193 → 214 (+21)**. These values do not match the source-backed rows above. Their model, prompt, and exact export have not been established from the inspected package, so the LaTeX retains them as a clearly marked **prior draft / unresolved** row for supervisor reconciliation, without ranking them against the verified results. Do not assign them to Gemini or invent a prompt/version.

Before final submission, identify that original run and attach its source, or remove the unresolved row. The expanded verified table is otherwise usable now; exact prompt-text snapshots remain a reproducibility gap to fill.

## Reproduce and inspect

Run `python3 reconcile_proof_coverage.py` from this folder (standard library only). It is read-only and prints the audit JSON. `proof_coverage_audit.ipynb` provides an inspectable entry point; `proof_coverage_audit.json` records the full result. The CSV contains the 17 verified rows only; the original draft row is excluded.

The two standalone LaTeX snippets require `booktabs` and span both columns using `table*`. The manuscript includes them with `\\input{include/langpro-results-1000}` and `\\input{include/langpro-results-365}`. Updated prose uses source-backed counts; the gold-223 table and original JSONs are unchanged.
