# Flash-Lite problem-level contrasts

These files support qualitative inspection of Ettore's completed shared-protocol
Gemini 3.1 Flash-Lite run on 1,000 problems. LEX is prompt 1; Stefan is prompt 2.
They do not combine different experimenters' runs or include the 365 population.

| Unsolved system → solved system | LEX | Stefan |
|---|---:|---:|
| WordNet → LLM-only one-shot | 26 | 29 |
| WordNet → LLM-only agentic | 39 | 38 |
| WordNet+LLM one-shot → WordNet+LLM agentic | 14 | 16 |

Start with `index.json` for filenames and counts. Each contrast file includes
problem ID, premise, hypothesis, gold label, and two system objects. Each system
object contains its outcome and `kb`: the additional relations actually injected
for the selected attempt, or an empty list when the baseline was retained without
an additional KB. `attempts` preserves all recorded KB evaluations for that system.

**WordNet's internal relation set is not enumerated in these exports.** It is
represented by `wordnet_enabled` and `wordnet_internal_relations: null`; an empty
`kb` must not be interpreted as an empty WordNet knowledge base. The `kb` lists
come from prover inputs, not generated prose or unfiltered candidate relations.

One-shot uses baseline plus the first attempt of the same agentic execution.
Unknown outcomes remain distinct from neutral. Cases can overlap between files;
the counts are directional transitions, not differences between aggregate scores.
No semantic-validity annotations are assigned: these JSONs are evidence for manual
inspection, not a claim that every proof-enabling relation is valid lexical knowledge.

Source paths and SHA-256 hashes are recorded in each file (`sha256_lf`: source
bytes with CRLF normalized to LF, for portability across Git checkouts). Record-level provenance
points to the original workspace archive when available. Reproduce with
`python analysis/generate_contrasts.py` from the parent archive folder. This reads
existing results and writes only the analysis JSONs; it makes no API/prover calls.
