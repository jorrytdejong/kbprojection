# Gemini 3.1 Flash-Lite: Jorryt's LEX-prompt agentic pipeline

Jorryt confirms that these are his LEX-prompt agentic-pipeline results on 1,000 SNLI entailment problems, not Stefan's historical Table 4 results. Model: `google/gemini-3.1-flash-lite`, provider: OpenRouter. The exact prompt template revision is not recorded in these JSONs; do not infer byte-identical templates from the shared LEX label.

| Configuration | One-shot | Agentic | Gain | Source |
|---|---:|---:|---:|---|
| WordNet only | 162 | -- | -- | `wordnet-only.json` |
| LLM only (second export; table row) | 124 | 149 | +25 | `llm-only-second-run.json` |
| WordNet + LLM | 192 | 202 | +10 | `llm-and-wordnet.json` |
| LLM only (earlier export; version comparison) | 125 | 149 | +24 | `llm-only-first-run.json` |

One-shot counts successful `baseline_langpro` or `langpro_with_kb` attempt 1 events, with `output.matches_gold: true`. Agentic counts final `outcome.solved: true`. LLM-only skips baseline; WordNet+LLM retains successful baseline problems. Null baseline results are not success. `max_iterations: 2` is initial generation plus at most one refinement, not evidence against the agentic interpretation. Both saved LLM-only files contain both stages: their filenames do not mean one-shot versus agentic.

The earlier versus second export differs in first-attempt success for problem `7589467042.jpg#0r4e`; both have a successful final outcome for that problem and the same final solved-ID set. Do not average the exports or count them as independent replications without further provenance. The expanded table uses the second export and discloses the earlier value.

Every export has 1,000 unique IDs and gold label entailment throughout. Consequently, for these verified predictions, 124/1,000 and 149/1,000 correspond to 12.4% and 14.9% accuracy, and 192/1,000 and 202/1,000 to 19.2% and 20.2%. The previous note claiming that accuracy was necessarily a different metric was incorrect; raw timelines establish the stage-specific counts directly.

The other recent model exports use the same 1,000 IDs but a WordNet baseline of 157. Five baseline successes here are classified as unknown there. The shared protocol documents stricter resource-limit/conflicting-proof interpretation; this is not an exact replay of this export. The table includes Gemini under LEX while explicitly identifying this version difference.

See `../reconcile_proof_coverage.py`, `../proof_coverage_audit.json`, and `../TABLE_PROVENANCE.md` for checks and source mapping. No original JSON or workbook was changed.
