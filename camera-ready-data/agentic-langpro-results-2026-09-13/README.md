# LangPro evaluation results

Start with [the result tables](reports/tables.txt) or [the source and interpretation guide](reports/table-guide.md).

| Folder | Contents |
|---|---|
| `results/1000/` | Five completed models, both prompts, five configurations on the 1,000-problem population. |
| `results/365/` | Four completed models, including Flash Lite, both prompts, five configurations on curated365. |
| `reports/` | Aligned text tables, machine-readable metrics, proof-coverage CSV and the source guide. |
| `reports/latex/` | Separate paper tables for the 1,000- and 365-problem populations. |
| `protocol/` | Frozen problem lists and original scientific/execution configuration snapshots. |
| `reference/` | Jorryt's earlier Flash Lite run and the separate 223-problem gold oracle. |
| `audit/` | Verification script/notebook, recorded checks, branch status, recovery provenance and original archive notes. |

## Results and populations

Each current model directory contains `lex/` and `stefan/`, each with WordNet-only, LLM-only one-shot/agentic, and WordNet+LLM one-shot/agentic JSON files. There are **90 current configuration exports and 64,600 records**. Repeated configurations and prompts share underlying work; these are not 64,600 independent problems.

The 1,000 and 365 populations are distinct. Flash Lite appears in both, as two separate evaluations. The reference Jorryt run uses a different execution version and stays separate. Unknown, technical error and missing outcomes must not be converted to neutral.

## Main files

- [Tables](reports/tables.txt)
- [Metrics](reports/metrics.json)
- [Proof coverage](reports/proof-coverage.csv)
- [Table provenance and interpretation](reports/table-guide.md)
- [1,000-problem LaTeX table](reports/latex/langpro-results-1000.tex)
- [365-problem LaTeX table](reports/latex/langpro-results-365.tex)
- [Manifest](manifest.json)
- [Flash Lite curated365 reuse provenance](audit/flash-lite-365-provenance.json)

## Verification

From this archive root, run:

```sh
python audit/reconcile_proof_coverage.py
```

This reads saved data only; it makes no API or prover calls. The notebook in `audit/` runs the same checks. The oracle runner under `reference/gold-oracle-223/` is an archived original, not part of this verification command; it requires its original source environment and annotation input.

Problem-level exports are preserved byte-for-byte. Their embedded source-workspace paths, timestamps and configuration references are original provenance, not rewritten package paths. Use `manifest.json` for current package paths and [the path map](audit/path-map.json) for old-to-new filenames. Large raw prover responses remain in the original workspace rather than duplicated in this portable archive. Configuration snapshots in `protocol/` preserve their original scope; later changes and recoveries are documented in `audit/`.

The table with the extra prover-error column is retained in `audit/tables-with-prover-errors.txt`; the normal summary is `reports/tables.txt`.
