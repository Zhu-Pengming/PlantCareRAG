# Dataset v1 data-quality decision

## Decision

The rebuilt v1 uses only the two datasets whose Kaggle pages declare CC BY
4.0. The most apparently RAG-ready candidate is rejected because its license is
Unknown and the downloaded file does not match the advertised field list.

| Dataset | Actual downloaded content | Decision |
|---|---|---|
| Plants Growth and Care Recommendations | 596 CSV rows, 6 columns, CP1252 | KB seed after conflict exclusion |
| Indoor Plant Health & Growth Dataset | 1,000 CSV rows, 17 columns | Context stress test only after consistency filtering |
| Indoor House Plants Dataset with Care Instructions | 209 JSON rows, 15 fields | Rejected |

The exact URLs, archive hashes, member hashes, encodings, licenses, and roles
are frozen in `config/datasets.json`.

## KB seed audit

- 596 source rows contain 4 exact duplicates after whitespace and capitalization
  normalization.
- The remaining rows represent 387 normalized plant names.
- 124 names have conflicting values in at least one care field and are excluded
  as a whole; they are not resolved by majority vote.
- Conflicts affect watering for 112 names, fertilization for 96, soil for 85,
  growth for 59, and sunlight for 26.
- 263 conflict-free names remain. Each produces exactly five atomic entries:
  growth, soil, lighting, watering, and fertilizer. The final KB therefore has
  1,315 entries.
- The records provide common plant names, not verified botanical identities.
  The project must not imply species-level taxonomic precision.
- Every accepted fact is labeled `single_dataset_claim`; it is dataset-backed,
  not independently verified horticultural guidance.

Important exclusions include Peace Lily, Spider Plant, and Pothos because the
dataset gives them conflicting rows. Monstera, Snake Plant, and ZZ Plant pass
the mechanical consistency gate. Passing the gate means “internally
non-conflicting in this file,” not “botanically proven correct.”

## Context-data audit

The 1,000-row health dataset contains 350 direct pest-field contradictions:

- 155 rows say no pest is present but assign Low, Moderate, or High severity;
- 195 rows name a pest but set severity to None.

Those rows are excluded, leaving 650 context samples. Even retained samples are
not treated as causal or diagnostic ground truth: the dataset page does not
provide an auditable observation protocol, and mean health scores are nearly
flat across symptom and pest categories. Every retained row is explicitly
limited to `context_stress_test_only` and prohibited from supporting ground
truth or safety claims.

## Rejected candidate audit

The downloaded `house_plants.json` has 209 rows, not the advertised toxicity /
care-difficulty / description structure. It contains no such fields. The audit
also found:

- only 159 unique Latin-name strings for 209 rows;
- `insects` is a list in 135 rows and a string in 74, while `diseases` also
  mixes strings and lists;
- 76 maximum-temperature Celsius/Fahrenheit pairs fail unit conversion;
- taxonomic spelling problems such as `Aechmea fatsiata` and
  `Zamioculcas zamifolia`;
- an Unknown Kaggle license.

It is excluded from downloads by default and none of its values appear in the
processed corpus.

## Benchmark interpretation

The 1,315 questions are deterministic templates generated from the same schema
as the KB. They are useful only as a retrieval contract test. They do not
measure factual correctness, conversational robustness, symptom diagnosis, or
real-user performance.

On the frozen hash split, test N=658:

| Stage | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Raw BM25 | 26.4% | 66.9% | 51.9% |
| Entity + dimension candidate gate, then BM25 | 100.0% | 100.0% | 100.0% |

The first implementation misclassified four Water Lily questions because the
entity token “water” triggered the watering rule. The final pipeline links and
masks the plant name before classifying the dimension. This is a retrieval
plumbing fix, not a claim about a learned intent model.

## Runtime controls

The deployable query layer carries the audit decisions forward instead of
forgetting them after preprocessing:

- accepted entities can answer only the five source-backed care dimensions;
- all 124 quarantined names remain linkable so the system can return an
  explicit conflict refusal rather than pretending the plant is unknown;
- pet safety, disease/symptom, pest, temperature, humidity, and taxonomy
  questions are structurally refused;
- an unrecognized entity cannot silently fall through to arbitrary global
  search results;
- global discovery is allowed only when the wording explicitly asks for plant
  recommendations or a list;
- answered results expose evidence level, dataset page, license, and raw row
  numbers.

These controls reduce misuse risk but do not upgrade the underlying claims.
They remain single-dataset assertions with no independent horticultural
verification.
