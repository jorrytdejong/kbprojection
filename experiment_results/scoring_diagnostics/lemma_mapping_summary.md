# Lemma mapping audit

This audit applies the evaluation's contextual POS lemmatisation to the saved predictions. `lemma_mapping_audit.csv` has one row for every one of the 14,480 prediction slots and shows the raw and lemmatised relation sets. `lemma_argument_audit.csv` expands each argument occurrence and includes the sentence span and POS tags used by the scorer. These are scoring representations; the saved model outputs were not rewritten.

## How often did the prediction change?

There are 12,805 valid prediction slots, 1,637 non-entailment outputs, and 38 errors. The latter two groups are included in the 14,480-row file but are excluded from lemmatisation and scoring.

Of the 12,805 valid slots, 10,593 (82.7%) had no argument changed by lemmatisation, while 2,212 had at least one changed argument. The 10,593 includes 1,394 valid predictions with no relation arguments. Among the 11,411 valid predictions that do contain relations, 9,199 (80.6%) had no argument changed.

Across the 37,703 argument occurrences in valid predictions:

| Outcome | Count | Share | Meaning |
| --- | ---: | ---: | --- |
| Changed | 3,532 | 9.4% | A sentence span was found and WordNet returned a different lemma. |
| Matched, unchanged | 21,192 | 56.2% | A unique mapped POS sequence was found, but lemmatisation returned the same form. |
| Absent | 12,711 | 33.7% | The argument was not found as an exact contiguous token span in either sentence. |
| Ambiguous | 267 | 0.7% | Matching occurrences yielded conflicting mapped POS sequences. |
| Empty | 1 | <0.1% | The argument was empty. |

Thus, “no change” does not mean the tagger failed in 82.7% of predictions. Many arguments were already in a WordNet lemma form, and 1,394 valid predictions had no relations to transform. Conversely, 33.7% of argument occurrences were absent from the source sentences and could not be assigned contextual POS by this procedure. Absence is an exact-span coverage limitation; it does not by itself establish a POS-tagging error. The audit does not compare tags with gold POS labels, so it cannot estimate tagger accuracy.

## Examples

- **Changed:** `standing in line` becomes `stand in line`. The premise contains `standing in line`, tagged `VBG IN NN`; the verb tag maps to WordNet's verb category. `waiting` in the hypothesis is tagged `VBG` and becomes `wait`.
- **Matched, unchanged:** `hammer` is found in the premise and hypothesis as `NN` and remains `hammer`. `wooded area` is found as `JJ NN` and remains unchanged because WordNet returns the same forms.
- **Absent:** With premise “A band is performing onstage” and hypothesis “A band is playing onstage,” the predicted arguments `perform` and `play` are not exact surface spans. The scorer leaves them as written. This case illustrates the exact-span limitation; since they are already base forms, it does not necessarily prevent the two forms from matching after scoring.
- **Ambiguous:** `concrete` is tagged `JJ` in the premise and `NN` in the hypothesis. Because those occurrences imply different WordNet POS categories, the scorer leaves `concrete` unchanged.

For the supervisor's example, in “A person is walking” / “A person is hiking,” both arguments are tagged `VBG` and map to `walk` and `hike`, respectively. The relation remains directed: `walk < hike`. Lemmatization normalizes inflection; it does not determine that *walking* and *hiking* are synonymous or equivalent.

## How to inspect the CSVs

In `lemma_mapping_audit.csv`, filter `status` to `valid` and sort by `changed_argument_occurrences` to find predictions whose mapping changed. The per-argument count columns distinguish changed, already-unchanged, absent, ambiguous, and empty arguments. `mapping_details_json` gives each argument's original text, lemma, decision, and POS evidence.

In `lemma_argument_audit.csv`, filter `decision` to `absent` or `ambiguous` to inspect why an argument was left as written. `POS_evidence_json` is empty for absent arguments and lists the matching sentence spans and tags for located arguments. Repeated occurrences across models and runs are counted separately in the occurrence totals. Relation argument direction is preserved in both exports.
