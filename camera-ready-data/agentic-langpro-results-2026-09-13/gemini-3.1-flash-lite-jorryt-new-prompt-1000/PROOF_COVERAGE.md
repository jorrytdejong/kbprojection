# Gemini 3.1 Flash-Lite: Jorryt's new-prompt run

These are Jorryt's own results on the 1,000-problem SNLI entailment population, using a new prompt. They are not Stefan's historical Table 4 results. The exact prompt template and version are maintained outside this archive.

Proof coverage below counts records whose JSON `outcome.solved` value is `true` (the WordNet-only export also records the total in `config.solved`). The workbook's `Acc` values are a different metric and should not be substituted for proof counts.

| Configuration | Source JSON | Proofs / 1,000 |
|---|---|---:|
| WordNet only | `wordnet-only.json` | 162 |
| LLM only, saved run 1 | `llm-only-first-run.json` | 149 |
| LLM only, saved run 2 | `llm-only-second-run.json` | 149 |
| WordNet + LLM | `llm-and-wordnet.json` | 202 |

The LLM-run configs specify `max_iterations: 2`. The exports therefore should not be relabeled as “one-shot” or “agentic” without checking the external prompt/protocol metadata. The two LLM-only files are separately saved runs, not independent prompt arms.
