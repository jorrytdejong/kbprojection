# Knowledge Base PROver inJECTION (kbprojection)

This library is designed to facilitate the use of Large Language Models (LLMs) to generate Knowledge Base (KB) injections for the LangPro prover. It provides tools for prompting LLMs, processing the generated KBs, and orchestrating experiments to evaluate the effectiveness of these injections.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or install the project into an existing environment
uv pip install -e .
```

`uv sync` installs the project and its dependencies into `.venv`. Use Python
3.10 or newer; the repository's type annotations require it.

## Runtime configuration: local or Google Colab

The package can run with local paths or Drive-backed Colab paths through one
shared runtime helper. The switch is controlled by environment variables:

* `KBPROJECTION_RUNTIME`: `local` or `colab`. If unset, Colab is auto-detected.
* `KBPROJECTION_PROJECT_ROOT`: base project directory.
* `KBPROJECTION_DATA_DIR`: optional override for datasets.
* `KBPROJECTION_CACHE_ROOT`: optional override for experiment JSON caches.
* `KBPROJECTION_RESULTS_DIR`: optional override for CSV/JSONL outputs.
* `KBPROJECTION_THIRD_PARTY_CACHE`: optional override for NLTK, Hugging Face,
  sentence-transformers, and Torch caches.

Local usage usually needs no extra setup:

```python
from kbprojection.runtime import configure_runtime

paths = configure_runtime()
DATA_DIR = paths.data_dir
CACHE_ROOT = paths.cache_root
RESULTS_DIR = paths.results_dir
```

In Google Colab, mount Drive and point the project root at your copied repo:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/kbprojection")
os.environ["KBPROJECTION_RUNTIME"] = "colab"
os.environ["KBPROJECTION_PROJECT_ROOT"] = str(PROJECT_ROOT)
```

Then install and configure:

```python
%cd /content/drive/MyDrive/kbprojection
%pip install -r requirements-colab.txt -e .

from kbprojection.runtime import configure_runtime
paths = configure_runtime(project_root=PROJECT_ROOT)
```

This keeps datasets, experiment caches, result files, NLTK data, Hugging Face
models, sentence-transformer models, and Torch cache files under the project
root instead of Colab's temporary VM storage. See
`kbprojection_colab_setup.ipynb` for a ready-to-run Colab bootstrap notebook.

## Usage

The library is divided into several modules:

* `kbprojection.loaders`: Data loaders for SNLI and SICK datasets.
* `kbprojection.models`: Pydantic models for type safety across the pipeline.
* `kbprojection.prompts`: Manage and fill prompt templates.
* `kbprojection.langpro`: Interface with the LangPro API.
* `kbprojection.llm`: A unified interface for calling various LLMs (OpenAI, Anthropic, Gemini).
* `kbprojection.filtering`: Functions to normalize and filter the generated KB injections.
* `kbprojection.orchestration`: High-level functions to run experiments.

### Automatic Data Downloading

The dataset loaders (`SNLILoader` and `SICKLoader`) automatically download the necessary data if it is not found in the specified directory. Without an explicit directory, they use a dataset subdirectory under `KBPROJECTION_DATA_DIR`, or the persistent application data directory when that variable is unset.

### Example: Loading a Single Problem

```python
from kbprojection import SNLILoader
from kbprojection.runtime import configure_runtime

paths = configure_runtime()

# Initialize loader pointing to your data directory
# Ensure data is downloaded (runs automatically if not present)
loader = SNLILoader(data_dir=paths.data_dir / "snli")

# Get a specific problem by ID (e.g., from SNLI dev set)
problem = loader.get_problem("4705552913.jpg#2r1n", split="dev")

print(f"Problem ID: {problem.id}")
print(f"Premises: {problem.premises}")
print(f"Hypothesis: {problem.hypothesis}")
print(f"Gold Label: {problem.gold_label}")
```

### Example: Full Experiment Orchestration

```python
import asyncio
from kbprojection import collect_kb_helpful_examples_random, SNLILoader
from kbprojection.models import ProblemConfig, TestMode
from kbprojection.runtime import configure_runtime

paths = configure_runtime()

# Initialize dataset loader
# If data is not present, it will be downloaded automatically.
snli_data = SNLILoader(data_dir=paths.data_dir / "snli")

# Configure the experiment
config = ProblemConfig(
    llm_provider="openai",
    model="gpt-4o",
    prompt_style="icl",
    test_mode=TestMode.BOTH,  # Test both raw LLM KB and filtered KB
    run_ablation=False,       # Set to True to find minimal set of injections
    verbose=True
)

async def main():
    async for res in collect_kb_helpful_examples_random(
        dataset=snli_data,
        config=config,
        split="dev",
        label_filter={"entailment", "contradiction"},
        max_matches=1,
        max_checked=10,
        cache_dir=paths.cache_root / "readme_example"
    ):
        print(f"Problem {res.problem.id}: Fixed with KB: {res.kb_filtered}")

asyncio.run(main())  # In a notebook, use: await main()
```

### Example: Manually Creating and Executing a Problem

You can also create a problem instance manually and process it through the pipeline.

```python
from kbprojection.models import NLIProblem, NLILabel, ProblemConfig
from kbprojection import run_problem

# 1. Create a manual problem
manual_problem = NLIProblem(
    id="manual-test-1",
    premises=["A dog is running in the park."],
    hypothesis="An animal is moving.",
    gold_label=NLILabel.ENTAILMENT,
    dataset="manual",
    split="test"
)

# 2. Process the problem
# This runs the full pipeline: No-KB check -> LLM generation -> Filtering -> Re-check
config = ProblemConfig(
    llm_provider="openai",  # or "anthropic", "gemini"
    model="gpt-4o",
    verbose=True
)

result = run_problem(manual_problem, config=config)

print(f"Final Status: {result.final_status}")
if result.kb_filtered:
    print(f"Generated KB: {result.kb_filtered}")
```

## Core Models

### NLIProblem

Represents a single NLI problem instance.

* `id`: Unique identifier for the problem.
* `premises`: A list of premise sentences.
* `hypothesis`: The hypothesis text.
* `gold_label`: The ground truth label (`entailment`, `contradiction`, or `neutral`).
* `dataset`: Source dataset name (e.g., "snli", "sick").
* `split`: Dataset split key (e.g., "train", "dev", "test").
* `original_data`: Dictionary containing original raw data from the dataset wrapper.

### ProblemConfig

Configuration object for the pipeline.

* `llm_provider`: String identifier for the LLM provider (e.g., "openai").
* `model`: Model identifier (e.g., "gpt-4o").
* `prompt_style`: Identifier for the prompt template style.
* `filtering`: `FilteringConfig`; shared transformation settings, described below.
* `post_process`: Legacy optional boolean for underscore/preposition cleanup only. `False` disables those two operations; it does not disable lemmatization or argument alignment. Conflicting explicit cleanup settings are rejected.
* `test_mode`: `TestMode` enum controlling which stages to run (`no_kb`, `raw_kb`, `normalised`, `both`, `full`); `filtered` is a deprecated alias for `normalised`.
* `run_ablation`: Boolean; if `True`, runs ablation to find all minimal sufficient KB subsets.
* `verbose`: Boolean; enables detailed logging.

### Configurable KB filtering

Use one configuration object for the operational pipeline and offline F1 evaluation.
The operational default is now **contextual POS additive lemmatization**. Other
operational transformations retain their historical defaults. Select
`lemmatization_kind="verb"` explicitly to recover the historical verb-additive profile.

```python
from kbprojection import FilteringConfig, ProblemConfig, pipeline_filter_kb_injections

config = ProblemConfig(filtering=FilteringConfig.operational())
results = pipeline_filter_kb_injections(
    ["isa_wn(dogs, animals)"],
    ["Two dogs are running."],
    "Two animals are moving.",
    filtering=config.filtering,
)
```

Pass `config=config` to the public runners. Keep new filtering settings inside
`ProblemConfig.filtering`; do not also pass the legacy `post_process` runner argument.

| Setting | Operational default | Evaluation baseline |
|---|---|---|
| `underscores_mode` | `replacement` | `replacement` |
| `leading_preposition_mode` | `replacement` | `replacement` |
| `lemmatization_kind` | `pos` | `pos` |
| `lemmatization_mode` | `additive` | `off` |
| `diff_only_mode` | `additive` | `off` |
| `argument_alignment_mode` | `additive` | `off` |
| `lemma_match_policy` | `exact_or_lemma` | `exact_or_lemma` |
| `final_ph_filter` | `True` | `False` |
| `use_semantic` | `False` | `False` |

For candidate-producing transformations, `off` leaves the input unchanged without
calculating variants; `additive` retains the input and adds distinct variants;
`replacement` retains only the transformed candidate when applicable and otherwise
keeps the input. Underscore cleanup replaces `_` with spaces. The leading-preposition
rule removes the first token only from a three-token argument whose first POS tag is
`IN`; this can change argument content. Diff-only extracts differing corresponding
tokens from equal-length multiword arguments.

Contextual POS lemmatization looks for the argument as a contiguous token span in
its assigned sentence (argument 1 in P, argument 2 in H). It uses noun, verb,
adjective and adverb tags where available. If no span is found, or matching spans
have conflicting tag sequences, it keeps the argument unchanged. This span lookup
is distinct from the token-membership check used for P/H matching.

Morphological matching is a separate policy: `exact`, `exact_or_lemma`, or
`lemma_only`. It checks word presence using exact tokens and/or WordNet noun and
verb lemmas; it does not generate relations or expand WordNet synonyms/hypernyms.
Multiword presence uses token membership, not contiguous phrase matching. Turning
KB lemmatization off does not implicitly turn morphological matching off.
Semantic matching remains an optional final check, requires `final_ph_filter=True`,
and is unavailable in offline F1 replay.

`argument_alignment_mode="additive"` adds reversed candidates, preserving the
predicate; the final strict P/H filter, when enabled, retains candidates whose first
argument matches the premise and whose second argument matches the hypothesis.
`replacement` reorients a pair only when the reverse direction matches and the
original direction does not. Already aligned, ambiguous, or unsupported pairs remain
unchanged. Audit records explain each decision. Lexical alignment does not establish
the semantic validity of the reversed relation.

For example (illustrative, not a dataset result), with premise “Emma is cleaning the
room” and hypothesis “Emma is tidying up the room”, replacement can transform
`isa_wn(tidy up, clean)` into `isa_wn(clean, tidy up)`. It changes the prediction;
it does not make those two directed pairs equivalent in scoring.

Text normalization runs first. Under additive operational settings, lemma and
diff-only variants derive independently from the normalized base, followed by
additive argument reversal. If either lemma or diff-only uses replacement, they run
sequentially (lemma then diff) on current candidates, so replaced originals cannot
reappear through a sibling branch. Argument alignment in replacement mode runs
before these stages, allowing POS tagging to use the corrected P/H direction.
CLI flag order never changes this order. `KBResult.transformations` and
`alignment_reason` expose the transformation history and alignment decision.
Syntax/predicate acceptance, identical-argument suppression and stable deduplication
remain common pipeline controls, including when the final P/H filter is off. Their
effects are recorded separately in the replay audit.

The legacy `filter_kb_by_prem_hyp` wrapper only filters existing candidates; its
unused `swap_args` option is deprecated. Use the main pipeline for transformations.
Complete operational results store resolved settings and pipeline version; their
cache keys include the input and configuration. Historical cache records without
matching metadata are not silently reused.

### Diagnose F1 with different scoring rules

#### Reproducibility

The current diagnostic question is how much KB scores depend on inflection and
argument direction. Keep the saved predictions and annotations unchanged and
vary only their scoring representations:

```bash
uv sync
uv run python -m nltk.downloader -d .cache/nltk punkt_tab averaged_perceptron_tagger_eng wordnet
uv run python scripts/experiments/evaluate_kb_scoring.py --compare
```

The NLTK setup is needed once for lemma scoring. Baseline and argument-order-only
scoring need no NLP resources. For proxy/manual setup, see the resource instructions
below. Replay performs no downloads, LLM calls or LangPro calls.

| Scoring condition | Lemmatize both sides | Ignore binary argument direction |
|---|---|---|
| Original | No | No |
| Lemmas | Yes | No |
| Argument order agnostic | No | Yes |
| Combined | Yes | Yes |

The printed comparison and report contain one table per scoring condition, with
exact match, micro-F1, micro-precision and micro-recall. Values are percentages as
mean (± sample SD) across generation runs, rounded to one decimal place; models
are sorted by descending mean micro-F1. Exact match requires the complete scored
KB to equal a selected reference under that rule. The original condition is kept
alongside the three diagnostic conditions for comparison.

The command prints this four-condition comparison. The repository keeps the compact
[report](experiment_results/scoring_diagnostics/REPORT.md),
[`summary.csv`](experiment_results/scoring_diagnostics/summary.csv), and
[`manifest.json`](experiment_results/scoring_diagnostics/manifest.json). The manifest
records source, input, NLP-resource, and generated-output hashes; `code_sha256`
identifies the source bytes even when an amend or squash changes a Git commit ID.

With no flags the command prints the original baseline. Select a single rule with
`--lemmatize` or `--argument-order-agnostic`, or combine both flags. `--compare`
selects all four rules and cannot be mixed with individual flags. Printing requires
no output path or JSON input. To export detailed CSVs, configurations, examples and
a manifest, choose a fresh output directory:

```bash
uv run python scripts/experiments/evaluate_kb_scoring.py --compare --output .cache/scoring-run
uv run python scripts/experiments/evaluate_kb_scoring.py --help
```

#### NLTK resources

Installed NLTK data is discovered automatically through standard NLTK paths,
`NLTK_DATA`, the configured runtime cache (by default `.cache/nltk`), repository
`nltk_data`, and `nltk_data` beside the active environment directory. If a required
resource is missing, the error names it and gives a one-time installation command.

If a proxy prevents the downloader from connecting, follow the
[official NLTK proxy or manual installation instructions](https://www.nltk.org/data.html).
Alternatively, fetch the official archives in an environment with working access
and transfer them to the offline machine:

| Resource | Official archive | Location under `.cache/nltk/` |
|---|---|---|
| Tokenizer | [punkt_tab.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip) | Extract into `tokenizers/` |
| English POS tagger | [averaged_perceptron_tagger_eng.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip) | Extract into `taggers/` |
| WordNet | [wordnet.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip) | Keep as `corpora/wordnet.zip` |

The resulting paths must include `tokenizers/punkt_tab/english/` and
`taggers/averaged_perceptron_tagger_eng/`. Automatic discovery does not imply that
NLP data is bundled with `uv sync`.

#### Scoring rules

These rules extend the shared scorer in `calculate_multi_reference_f1.py`, whose
overview-CSV CLI also accepts `--lemmatize` and `--argument-order-agnostic`.
In that CLI, `--argument-order-agnostic` preserves Jorryt's interface: it adds a
diagnostic F1 and exact-match result alongside the directed primary score.
`--lemmatize` compares lemmas on both sides before either metric is calculated.
Lemma scoring requires `premise` and `hypothesis` columns for contextual POS.
The same lemma function searches both sentences for every predicted/reference
argument, independent of its position. Exact contiguous occurrences must agree
on the WordNet POS sequence; missing or ambiguous spans remain unchanged. This
conservative policy is recorded with decision counts in the report. It does not
claim to normalize every possible inflection.

Original scoring retains the legacy parser's conventions, including ignoring
predicate names, duplicate relations and relation-list order. There is no
filtering, P/H argument correction, underscore/preposition replacement or removal
of self-pairs. Argument-order scoring treats binary `(a,b)`/`(b,a)` as equivalent;
legacy other tuple arities retain their argument order. Combined scoring applies
lemmatization first, then sorts binary arguments. Both operations are applied to
predictions AND every reference, only for evaluation.

Each condition selects its best reference separately using the existing tie-break.
Micro counts are aggregated within each generation run, then the mean and sample
SD are reported across runs. Canonicalization may merge tuples and change
denominators. A score increase is not guaranteed and does not establish semantic
correctness or LangPro proof success. The
[scoring diagnostic results](experiment_results/scoring_diagnostics/REPORT.md)
are the current evaluation results.

### Run the tests

```bash
uv sync --group test
uv run --group test python -m pytest tests -q
```

The `test` dependency group supplies pytest, including subtest reporting. `uv sync`
alone installs runtime dependencies, not this test group. The suite uses fixtures
and service mocks; it does not require NLP downloads or live LLM/LangPro access.

For future generation runs, `scripts/experiments/run_repeated_multi_reference_experiment.py`
accepts `--filtering-config PATH` to a JSON file containing one settings object and records the resolved configuration and
pipeline version in output rows. Resume rejects incompatible or unlabelled historical
outputs; use a new output path instead of mixing configurations.

### ExperimentResult

Encapsulates the results of running the pipeline on a problem.

* `problem`: The `NLIProblem` instance being processed.

* `kb_raw`: List of raw KB strings generated by the LLM.
* `kb_filtered`: List of filtered/formatted KB strings ready for LangPro.
* `kb_details`: List of `KBResult` objects containing detailed provenance for each injection.

* `pred_no_kb`: Prediction from LangPro without any KB injection.
* `status_no_kb`: Status of the baseline step.

* `pred_with_raw_kb`: Prediction using the raw (unfiltered) KB.
* `status_with_raw_kb`: Status of the raw KB evaluation step.
* `pred_with_kb`: Prediction using the filtered KB.
* `status_with_kb`: Status of the filtered KB evaluation step.

* `final_status`: `ExperimentStatus` enum summarizing the overall outcome (e.g., `BASELINE_SOLVED`, `NORMALISED_KB_SOLVED`, `KB_NOT_SOLVED`).
* `fixed_by`: String indicating which KB version fixed the problem (`"raw_kb"`, `"normalised_kb"`, or `"both"`), or `None`.
* `essential_kb`: Best minimal sufficient KB subset (ranked by token count). If ablation was run and multiple KB entries are redundant, this contains the simplest subset that alone fixes the problem.
* `ablation_subsets`: List of all minimal sufficient subsets found during ablation. Each subset is a list of KB strings that independently can fix the problem.
* `ablation_results`: Dictionary mapping serialized tested subsets to their resulting label.

## Multi-reference KB experiment workflow

The paper experiments compare LLM-generated KB relations with multiple human
KB annotations. Run the following commands from this repository's root, where
the experiment scripts and `data/` directory are located. The `.venv/bin/python`
examples use a Unix environment; on Windows, use `.venv/Scripts/python.exe` or
`python` from the activated environment.

### Input data

The experiment input CSV must contain one row per NLI item. The runner uses:

* `ID`: stable item identifier. Keep this non-blank for real experiment rows.
* `premise`: premise sentence shown to the model.
* `hypothesis`: hypothesis sentence shown to the model.
* `gold_label`: NLI label, such as `entailment`, `neutral`, or
  `contradiction`.
* `dataset`: source dataset, for example `sick` or `snli`.
* `split`: source split, for example `train`, `dev`, or `test`.

The multi-reference evaluator also needs the human KB reference columns:

* `Alternative_KB`
* `Ettore_KB`
* `Jorryt_KB`
* `Lasha_KB`
* `Stefan_KB`

Blank KB cells mean the annotator did not provide an annotation. `NO_RELATION`
means an explicit annotation that no KB relation is needed. Do not convert
blank cells into `NO_RELATION`.

### Canonical experiment input

The reproducible 362-item input is committed at
`data/all_usable_items_362.csv`. It contains the quality-controlled SNLI/SICK
entailment problems and the five reference LEX annotation columns used by the
multi-reference evaluations. The file has 362 data rows and its SHA-256 is:

```text
7be06d326bdaff587368b06c86e4b28ee0d8e642fda3cbf676a5baffd816e77e
```

The experiment scripts validate the required columns and row count before
making model calls. This prevents accidentally running the paper evaluation
on a different or incomplete CSV.

### Run an LLM experiment

Always run a small live smoke test before a full model run:

```bash
.venv/bin/python run_multi_reference_llm_experiment.py \
  --input-csv "data/all_usable_items_362.csv" \
  --output-csv "llm_outputs_smoke.csv" \
  --provider openrouter \
  --prompts ettore lasha \
  --models openai/gpt-5.4 \
  --limit 2 \
  --write-every 1
```

Then run the full experiment:

```bash
.venv/bin/python run_multi_reference_llm_experiment.py \
  --input-csv "data/all_usable_items_362.csv" \
  --output-csv "llm_outputs_sonnet45_gpt54_gemini35flash_all_usable.csv" \
  --provider openrouter \
  --prompts ettore lasha \
  --models \
    anthropic/claude-sonnet-4.5 \
    openai/gpt-5.4 \
    google/gemini-3.5-flash \
  --write-every 1
```

The output file contains three columns per prompt/model pair:

* `*_raw_response`: exact provider response.
* `*_KB`: parsed KB relations used for scoring.
* `*_error`: API or parsing error.

Resume is enabled by default. If a run stops, rerun the same command with the
same output file. Existing KB/error cells are skipped.

### Repeated-run model evaluation

Use `scripts/experiments/run_repeated_multi_reference_experiment.py` when each
model should process every item multiple times. Unlike the standard experiment
runner, this script writes one row per item, prompt, model, and repetition. This
long format allows accuracy and output stability to be evaluated separately.

The following experiment reproduces the paper's intrinsic LEX-prediction
evaluation: it runs the improved Lasha prompt five times with eight models on
all 362 usable annotation items. `--no-filter-kb` is intentional: the paper
scores the relations explicitly produced by each model, without KB-projection
normalization or premise--hypothesis filtering.

```bash
.venv/bin/python scripts/experiments/run_repeated_multi_reference_experiment.py \
  --input-csv "data/all_usable_items_362.csv" \
  --sample-size 362 \
  --repeats 5 \
  --prompts lasha \
  --models \
    openai/gpt-5.4-mini \
    anthropic/claude-haiku-4.5 \
    google/gemini-3.1-flash-lite \
    openai/gpt-oss-20b \
    google/gemma-3-4b-it \
    anthropic/claude-sonnet-4.5 \
    openai/gpt-5.4 \
    google/gemini-3.5-flash \
  --reference-columns \
    Alternative_KB Ettore_KB Jorryt_KB Lasha_KB Stefan_KB \
  --temperature 0 \
  --concurrency 4 \
  --write-every-jobs 40 \
  --request-timeout 120 \
  --max-retries 2 \
  --no-filter-kb
```

This creates `362 x 8 x 5 = 14,480` model calls. The production prompt name
`lasha` refers to the original Lasha prompt with the selected precision
calibration added.

Generated files are written to:

```text
experiment_results/lasha_all362_5runs/
```

The files have distinct purposes:

* `consistency_sample.csv`: frozen copy of the exact evaluated input items.
* `consistency_outputs.csv`: raw response, parsed KB, error, and repetition for
  every model call.
* `consistency_metrics.csv`: repeatability statistics for each prompt-model
  combination.
* `consistency_f1_by_run.csv`: multi-reference F1 for every individual
  repetition.
* `consistency_f1_summary.csv`: mean, sample standard deviation, minimum, and
  maximum F1 across repetitions.

The consistency metrics include:

* `all_runs_identical_rate`: fraction of complete items for which every
  repetition produced the same KB.
* `mean_pairwise_kb_f1`: average KB similarity between every pair of
  repetitions.
* `no_relation_flip_rate`: frequency with which a model alternated between
  `NO_RELATION` and a non-empty KB.
* `mean_unique_kb_sets_per_item`: average number of distinct KB answers per
  item.
* `error_rate`: fraction of unsuccessful model calls.

Both F1 variants use the same multi-reference best-match procedure as
`calculate_multi_reference_f1.py`. For each item, the model prediction is
compared with every available non-blank human reference, and the best reference
is selected before TP, FP, and FN are accumulated.

With `--no-filter-kb`, the parsed model relations are scored as generated. Do
not remove this flag when reproducing the paper's intrinsic LEX results;
omitting it enables the operational KB normalization/filtering pipeline and
therefore evaluates a different condition.

The standard metric treats relations as an unordered set.

Blank predictions and errors are skipped and counted; they are not interpreted
as `NO_RELATION`.

The output is resumable. Reusing the same output path skips rows that already
contain a KB or an error. Requests are retried during their initial execution,
but persisted error rows are not automatically retried on a later resume.

Temperature zero is requested for all models, but OpenRouter may ignore it when
the selected model does not support temperature control. At the time of this
experiment, it was unsupported for GPT-5.4 and GPT-5.4 Mini. Even where
supported, temperature zero does not guarantee identical hosted-model output.

A completed five-run experiment is available in
[`experiment_results/lasha_all362_5runs`](experiment_results/lasha_all362_5runs).

### Reproduce the committed LEX prediction scores (no API calls)

The saved long-format raw responses for all 5 runs, the 362 annotated items,
and the expected score files are committed. Reparse the raw responses and
recompute the set-based scores with:

```bash
mkdir -p /tmp/kbprojection-lex-replay
.venv/bin/python scripts/experiments/recompute_repeated_no_filter_scores.py \
  --input-csv experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_outputs.csv \
  --sample-csv data/all_usable_items_362.csv \
  --output-csv /tmp/kbprojection-lex-replay/no_filter_outputs.csv \
  --metrics-csv /tmp/kbprojection-lex-replay/no_filter_stability.csv \
  --f1-metrics-csv /tmp/kbprojection-lex-replay/no_filter_f1_by_run.csv \
  --f1-summary-csv /tmp/kbprojection-lex-replay/no_filter_f1_summary.csv \
  --filtered-f1-summary-csv experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_f1_summary.csv \
  --comparison-csv /tmp/kbprojection-lex-replay/filtered_vs_no_filter.csv \
  --repeats 5

diff -u \
  experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_f1_by_run.csv \
  /tmp/kbprojection-lex-replay/no_filter_f1_by_run.csv
diff -u \
  experiment_results/lasha_all362_5runs/small_medium_lasha_all362_5runs_no_filter_f1_summary.csv \
  /tmp/kbprojection-lex-replay/no_filter_f1_summary.csv
```

Both `diff` commands should produce no output and exit with status 0. The
committed `*_no_filter_f1_by_run.csv` contains the precision, recall,
micro-F1, and exact-best-match scores for every model and repeat;
`*_no_filter_f1_summary.csv` contains their five-run mean and sample standard
deviation. This procedure makes no network or model API calls.

### Multi-reference scoring method

The evaluator works item by item:

1. Parse the model KB as a set of relations.
2. Compare it to every non-blank human KB reference.
3. Select the best-matching reference for that item.
4. Accumulate TP, FP, and FN over all items.
5. Compute micro-precision, micro-recall, and micro-F1.

`Exact best match` means the model's full KB set exactly equals at least one
available human KB reference.

### Argument-order-agnostic relation micro-F1

The primary relation-set metric keeps the direction of each lexical relation:
`isa_wn(apple, fruit)` and `isa_wn(fruit, apple)` are different, because the
direction carries entailment meaning. Use the argument-order-agnostic metric as
a diagnostic sensitivity analysis when you instead want a reversed pair to
count as a match.

For example, the following prediction and reference do not match under the
primary metric but do match under this diagnostic:

```text
Prediction: isa_wn(fruit, apple)
Reference:  isa_wn(apple, fruit)
```

Add `--argument-order-agnostic` to calculate both the directed primary score
and the relaxed diagnostic in one run:

```bash
.venv/bin/python calculate_multi_reference_f1.py \
  --csv "llm_outputs_sonnet45_gpt54_gemini35flash_all_usable.csv" \
  --reference-columns \
    Alternative_KB Ettore_KB Jorryt_KB Lasha_KB Stefan_KB \
  --argument-order-agnostic \
  --summary-csv \
    "multi_reference_f1_argument_order_agnostic_summary.csv"
```

The summary retains the directed `micro_f1` fields and adds
`argument_order_agnostic_precision`, `argument_order_agnostic_recall`, and
`argument_order_agnostic_micro_f1`, plus
`argument_order_agnostic_exact_match_rate`. Exact match is an all-or-nothing
per-item score: after canonicalizing each two-argument pair, the complete
predicted relation set must equal the selected best human reference. The best
available human reference is selected independently under the relaxed rule.
Repeated-run reports include the corresponding per-run counts and mean,
standard deviation, minimum, and maximum fields automatically.

### Calculate inter-annotator agreement

The paper's Table 2 reports post-adjudication agreement, not the initial JSON
submissions. Its source is the tracked export of the edited project sheet:
[`data/annotator_agreement/iaa_overview_edit.csv`](data/annotator_agreement/iaa_overview_edit.csv).
The raw individual submissions remain available in
[`data/annotator_assignments`](data/annotator_assignments), but they represent
the pre-adjudication stage and do not reproduce the Table 2 scores.

Recompute Table 2 without network or API calls with:

```bash
mkdir -p /tmp/kbprojection-iaa
.venv/bin/python scripts/experiments/recompute_paper_iaa_scores.py \
  --iaa-csv data/annotator_agreement/iaa_overview_edit.csv \
  --final-items-csv data/all_usable_items_362.csv \
  --summary-csv /tmp/kbprojection-iaa/paper_table2_iaa_scores.csv

diff -u experiment_results/iaa/paper_table2_iaa_scores.csv \
  /tmp/kbprojection-iaa/paper_table2_iaa_scores.csv
```

For every annotator, the script filters to the final 362 IDs, selects the
best-matching KB from the other three original annotators, and reports exact
match and relation-level micro-F1. `Alternative_KB` is excluded because it is
an adjudication-created fourth valid variant rather than an independent
annotator submission. The recomputed micro-F1 values match Table 2. Ettore's
exact rate is 205/271 = 75.645...%, which displays as 75.6% at one decimal
place; Table 2 prints 75.7%.

To analyze the initial, pre-adjudication submissions separately, run
`calculate_inter_annotator_agreement.py data/annotator_assignments`.

### Prepare the prompt-ablation comparison (no API calls)

The prompt ablation uses the 223 unanimous items from the pre-finalization
edited IAA overview. This is intentional: two of these items were excluded
later from the 362-item final paper dataset, so do not replace the source with
`data/all_usable_items_362.csv`.

```bash
.venv/bin/python prompt_engineering/run_lex_prompt_ablation_overnight.py \
  --output-dir /tmp/kbprojection-prompt-ablation \
  --prepare-only
```

This writes the derived `agreed_subset.csv` (223 items), `primary_sample.csv`,
the deterministic stability sample, and the exact prompt templates into the
chosen output directory. It makes no model or network calls. A later live run
uses the same command without `--prepare-only`; it requires configured API
credentials and may vary because hosted model outputs can change.
