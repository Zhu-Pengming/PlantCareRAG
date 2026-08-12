# Dataset-backed Plant RAG v1

This is the replacement v1. It stops hand-authoring a tiny botanical KB and
instead treats dataset ingestion, provenance, conflict isolation, and retrieval
contracts as the engineering problem.

## What is included

- 263 common-name plant entities accepted by a deterministic consistency gate;
- 1,315 atomic care entries across growth, soil, lighting, watering, and
  fertilizer;
- 124 conflicted plant names quarantined with their raw row locators;
- 650 filtered health/context samples that are never used as ground truth;
- 1,315 template-derived retrieval contract queries, split by unsalted SHA-256
  into dev 657 / test 658;
- a standard-library BM25 baseline and an entity + dimension candidate gate.
- a runnable query engine with conflict, unsupported-dimension, and unknown-
  entity refusal gates plus row-level citations.

The KB seed and context dataset are both CC BY 4.0. Attribution and the exact
transformations are in `ATTRIBUTION.md`; immutable source URLs and SHA-256
checksums are in `config/datasets.json`.

## Quality boundary

The project intentionally does **not** answer toxicity, disease, or symptom
diagnosis questions. The accepted source has no defensible fields for them.
Facts are labeled `single_dataset_claim`, not “authoritative horticultural
truth.” A plant with conflicting source rows is excluded rather than merged by
majority vote.

The third candidate dataset is not used: its Kaggle license is Unknown, its
actual downloaded schema does not match the advertised description, and its
file contains severe consistency errors. See `reports/data_quality.md`.

## Baseline

The benchmark is deliberately named a **template contract test**. It verifies
that the retrieval plumbing selects the correct atomic field; it is not a
real-user QA benchmark.

| Test N=658 | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Raw BM25 | 26.4% | 66.9% | 51.9% |
| Entity + dimension gate, then BM25 | 100.0% | 100.0% | 100.0% |

At this scale the test caught an entity/intent collision that the old three-
plant KB could not expose: `Water Lily` initially triggered the watering rule.
The final pipeline masks the linked entity phrase before dimension routing.

## Ask it a question

The runtime deliberately answers only growth, soil, lighting, watering, and
fertilizer questions. Every result includes the dataset page, CC BY license,
and raw CSV row number. It refuses unsupported safety/diagnosis questions and
plants quarantined by the conflict audit.

```bash
python3 dataset_v1/scripts/ask.py \
  "When should I water Snake Plant?"

python3 dataset_v1/scripts/ask.py \
  "Which plants need indirect sunlight?" --top-k 3

python3 dataset_v1/scripts/ask.py \
  "Is Peace Lily toxic to cats?" --json
```

Response reason codes distinguish `supported_entity_care`,
`supported_discovery`, `conflicted_entity`, `unsupported_dimension`,
`unknown_entity`, and `ambiguous_intent`. Refused CLI requests exit with code
2, while answered requests exit with code 0.

## Reproduce

Python 3.9+ and the standard library are sufficient.

```bash
python3 dataset_v1/scripts/download_datasets.py
python3 dataset_v1/scripts/build_dataset.py
python3 dataset_v1/scripts/validate_dataset.py
python3 dataset_v1/scripts/evaluate_baseline.py
python3 -m unittest discover -s dataset_v1/tests -v
```

Raw downloads are ignored by Git. The download step verifies both archive and
extracted-member SHA-256 values before the build runs. Processed outputs and the
baseline result are committed so validation and tests do not require network
access.

## Structure

```text
dataset_v1/
  config/datasets.json      # source, license, role, frozen hashes
  data/processed/           # plants, entries, contract queries, contexts
  reports/                  # build counts, conflicts, quality decision
  results/baseline.json
  schemas/
  query_engine.py           # runtime linking, routing, retrieval, refusal
  scripts/                  # download, build, validate, evaluate, ask
  tests/
```

GitHub Actions repeats compilation, validation, tests, artifact checksums,
baseline byte-for-byte reproduction, and a query-engine smoke test without
downloading models or installing dependencies.

The previous hand-built experiment remains recoverable at Git tag
`v1-frozen`; it is not mixed into this dataset-backed result.
