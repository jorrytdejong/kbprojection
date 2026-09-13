# Lemma mapping audit

Generated from the saved `original` rows in `predictions.csv` with the same contextual POS scoring function used by the evaluation. `lemma_mapping_audit.csv` has one row per prediction slot; `lemma_argument_audit.csv` expands every relation argument and records matching sentence POS evidence.

- Prediction slots: 14,480 (12,805 valid; 1,637 non-entailment; 38 errors).
- Valid slots with no argument changed: 10,593 / 12,805 (82.7%); with at least one changed argument: 2,212.
- Valid slots with no relation arguments: 1,394.
- Argument occurrences: 37,703; changed: 3,532; matched to a unique POS sequence but unchanged: 21,192; absent from premise/hypothesis: 12,711; conflicting POS sequences: 267; empty: 1.
- Unique predicted (premise, hypothesis, argument) values: 3,059; changed: 626; unchanged: 1,271; absent: 1,129; ambiguous: 32; empty: 1.

At slot level, “no argument changed” means that none of the predicted relation arguments was lemmatised differently. It includes arguments absent from the sentences or ambiguous in POS, which are separately counted above. Non-entailment and error slots are included in the 14,480-row audit but have no lemmatised KB. Relation order is sorted in the exported representation for readability; argument direction is preserved.
