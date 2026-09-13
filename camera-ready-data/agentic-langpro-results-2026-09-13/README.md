# LangPro results for sharing

Historical Stefan Table 4 files contain the exact 1,000-problem population and are copied byte-for-byte. Both LLM-only versions are retained; they are not independent replications. The historical WordNet+LLM file includes 190 credit errors. These historical exports do not establish prompt/version equivalence with current runs.

Current exports use the same `config` + `records` JSON envelope. Agent records use `problem`, `outcome`, and `timeline`; WordNet-only records use Stefan's flat structure. Each model and initial prompt has five configuration files. Only completed branches are included in LLM configuration files; completed technical failures remain included and explicitly labelled. Missing records are omitted, counted in config/manifest, and listed by state in branch_status_snapshot.json. Baselines include all available baseline results.

The four completed current 1,000-problem models and the completed Gemini 365-problem model are distinguished by folder. GPT and Sonnet are complete in this snapshot. Flash Lite from the colleague is not imported as a new current run; the available older Stefan files remain historical.

This is schema adaptation, not a new scientific run. Unknown/resource-limit outcomes are not neutral. Correctness and categories are copied from the authoritative result ledger and checked against saved evaluations. One-shot timelines exclude critic/refinement; one-shot predictions are taken directly from the first-attempt evaluation, including empty-KB fallback. Normalisation candidate details are included when archived; no historical transformations or critic decisions are invented. Compact proof information replaces historical textual proof excerpts. `final_status` is the explicit current outcome category, not an invented historical stop code.

Phase references identify source journal requests and reuse. Large raw prover responses are intentionally not duplicated; source_result and request keys refer to the original workspace archive, not files inside this ZIP. This is a portable analysis export, not a full raw-response backup. No API keys or authorization headers are exported. No files have been uploaded to Drive.

## Authorized repair supplement
All current runs are complete. Six curated365 problems use archived autonomous local CCG, with exact unchanged text verified; see repair_manifest.json for parser provenance and hashes. Other cases retain remote CCG. Four local worker failures were recovered with identical requests and original remote CCG. Dependent agentic stages were resumed under the same scientific protocol. Original affected ledger results are preserved in repair_original_results.json. This replaces the earlier partial snapshot; LP errors, if any, remain explicit in final_tables.txt.

The current Flash Lite folder contains shared WordNet baselines only; the colleague's LLM run is not available locally. Historical Stefan files remain separate.
