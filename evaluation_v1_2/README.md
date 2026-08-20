# v1.2 — answerability and evidence contracts

This layer prepares PlantCareRAG for a real-query RAG evaluation without
changing or relabeling the frozen dataset-backed v1 benchmark.

## What this milestone adds

- multi-label care-dimension routing;
- qualifier extraction for evidence-sensitive conditions such as season;
- explicit insufficient-evidence, clarification, unsupported, and conflict outcomes;
- claim-to-evidence bindings for every answered factual claim;
- schemas for reviewed evidence, real-query annotations, and runtime responses;
- six authored smoke-test cases explicitly excluded from benchmark reporting.
- a finalized 10-plant, 38-claim authoritative-source overlay.

It does **not** add embeddings or LLM generation. Those components should be
evaluated only after a frozen real-query set and reviewed evidence overlay exist.

## Run it

~~~bash
python3 evaluation_v1_2/scripts/validate_contract.py
python3 evaluation_v1_2/scripts/validate_evidence_overlay.py
python3 evaluation_v1_2/scripts/review_evidence.py --check
python3 evaluation_v1_2/scripts/evaluate_contract.py
python3 evaluation_v1_2/scripts/ask.py "Should I fertilize Snake Plant during winter?"
python3 -m unittest discover -s evaluation_v1_2/tests -v
~~~

Build the observed-query review pool separately:

```bash
python3 evaluation_v1_2/scripts/download_query_sources.py
python3 evaluation_v1_2/scripts/build_query_candidates.py
python3 evaluation_v1_2/scripts/validate_query_candidates.py
python3 evaluation_v1_2/scripts/auto_triage_queries.py
python3 evaluation_v1_2/scripts/validate_query_shortlist.py
python3 evaluation_v1_2/scripts/screen_query_candidates.py --plant zz_plant --reviewer "Tom"
python3 evaluation_v1_2/scripts/screen_query_candidates.py --summary
```

Screening decisions are stored separately in `data/query_screening.json`; the
frozen candidate text is never edited. `selected_for_annotation` means only
that a question should receive manual labels. It does not make the question
benchmark-eligible.

Automated triage reduces the 399 observed candidates to a balanced 100-question
review shortlist (10 per entity). It predicts dimensions, mapping risks,
context requirements, answerability, and possible verified evidence. These
fields are explicitly marked `automation_only`; possible evidence IDs are not
gold labels. The interactive screener reviews the shortlist by default. Pass
`--all` only when auditing candidates excluded from the shortlist.

The shortlist is a deliberately balanced evaluation design, not a random
sample of Gardening Stack Exchange. It must not be used to estimate production
question prevalence. Ranking favors explicit entity mentions, answered posts,
care-dimension signals, and text-complete questions; the frozen rules and
output hashes are recorded in `config/query_triage_rules.json` and
`reports/query_auto_triage_stats.json`.

## Fully automated synthetic benchmark

The default no-human-review path is an 80-question, evidence-conditioned
synthetic contract benchmark:

```bash
python3 evaluation_v1_2/scripts/generate_synthetic_benchmark.py
python3 evaluation_v1_2/scripts/validate_synthetic_benchmark.py
python3 evaluation_v1_2/scripts/evaluate_synthetic_benchmark.py
```

It contains 66 single-dimension paraphrases, eight multi-dimension questions,
four unsupported pet-safety cases, and two unresolved common-name mapping
cases. Gold evidence is complete by construction because each broad
plant/dimension question is generated from every verified claim for that pair.
No human approval is required.

The benchmark deliberately separates direct wording (`v1`), novice wording
with an explicit dimension cue (`v2`), and frozen implicit paraphrases (`v3`).
The first baseline scores 100% on entity linking and on the direct, novice,
multi-dimension, refusal, and clarification groups. Implicit paraphrases expose
a real lexical routing gap: dimension/evidence exact match is 7/22 (31.8%) for
`v3`, producing 81.25% exact match overall and 84.8% evidence micro-recall.

This is a regression and component benchmark, not evidence of real-user
performance: the questions were generated from the same evidence they test.
The observed Stack Exchange candidate pipeline remains available only when
external-validity evaluation is wanted later.

## Benchmark boundary

The query annotation seed contains authored contract examples, not observed
user questions. Every record is marked as authored and benchmark-ineligible.
The real-query candidate pool comes from the frozen Gardening & Landscaping
Stack Exchange public data dump. Candidates retain the immutable Post ID,
canonical URL, raw wording, author attribution, creation time, and per-post
license. The reproducible build currently yields 399 unique candidates across
all 10 overlay entities (minimum 19 alias-matched candidates per entity).
Deterministic alias matching only selects records for review: every candidate
remains `pending_human_annotation` and benchmark-ineligible until a person
confirms relevance, plant identity, dimensions, answerability, and gold
evidence. Split assignment uses the stored unsalted SHA-256 of the stable ID.
See [`QUERY_SOURCE_ATTRIBUTION.md`](QUERY_SOURCE_ATTRIBUTION.md) and
`reports/query_candidate_stats.json`.

## Source authority is claim-specific

There is intentionally no global source tier:

| Claim type | Appropriate primary authority |
|---|---|
| horticultural care | Extension, RHS, botanical gardens |
| taxonomy | GBIF, Kew |
| pet safety | ASPCA, veterinary poison resources |
| current Kaggle corpus | dataset seed only |

The evidence schema records this with source type and authority purpose.
Conflicting claims remain separate evidence records; a later resolver must not
silently merge them.

## First evidence overlay

The overlay covers 10 accepted entities with 38 care claims from 10 NC State
Extension Plant Toolbox pages. It began with 24 claims; human review split
mixed-scope, mixed-season, and mixed-dimension claims, recovered directly
supported soil claims, and added source-stated lighting consequences.
Coverage remains intentionally uneven: a claim is included only when the plant
page states it directly.

Human review finalized 34 claims as `human_verified` and rejected four; no
claim remains pending. The rejected records are the Monstera and Hoya lighting
and watering mappings because the dataset's bare common/genus names do not
establish unique mappings to *Monstera deliciosa* or *Hoya carnosa*. Rejected
records remain in the overlay as auditable negative decisions but are excluded
by the runtime. The completed manual pass is tracked in
[`REVIEW_CHECKLIST.md`](REVIEW_CHECKLIST.md).

Review interactively without hand-editing JSON:

```bash
python3 evaluation_v1_2/scripts/review_evidence.py
python3 evaluation_v1_2/scripts/review_evidence.py --source dracaena
```

Each decision is saved atomically and appended to
`data/review_audit.jsonl`. Risky genus/common-name scopes require a second
explicit confirmation. `VerifiedEvidenceStore` is the runtime boundary: it
never returns pending or rejected records.

### Atomicity and qualifier policy

- A qualifier records only an applicability limit stated by the source; it is
  never inferred merely from surrounding page context.
- `season` means the claim applies within that season; `exception` means the
  claim applies generally except for the named condition. They are not interchangeable.
- Qualifier keys and values are closed vocabulary in all v1.2 schemas:
  `environment={indoor,landscape}`,
  `season={winter,spring,summer,autumn,spring_to_autumn}`,
  `exception={winter_dormancy}`, and
  `condition={low_light,direct_sun,cold_water,overwatering}`.
  New concepts require an explicit schema revision rather than an ad-hoc synonym.
- `environment`, `season`, and `exception` are applicability constraints and may
  be used as hard retrieval filters. `condition` restates the condition described
  by a claim and is a non-filterable controlled tag; it may inform ranking but
  must not exclude otherwise applicable evidence.
- One claim has one qualifier scope. A general rule and a seasonal rule must be
  separate claims even when the source places them in the same sentence.
- NC State prose care claims and structured Cultural Conditions can have
  different scopes. Structured-field claims must be marked `environment: landscape`
  unless the page explicitly supports a broader scope.
- Any content or qualifier edit invalidates the prior approval and resets the
  claim to pending. The audit log retains the earlier decision and revision.

## Frozen experiment protocol

The evidence and automated benchmark milestones are complete. Semantic-routing
development followed this frozen protocol:

1. Keep `synthetic_benchmark.json` and its SHA-256 frozen.
2. Add semantic or LLM dimension routing using dev only.
3. Report the final test split once, alongside the direct/implicit breakdown.

## Semantic dimension fallback result

The next experiment is complete. It keeps lexical routing whenever a dimension
cue is present and invokes a dense semantic fallback only when lexical routing
returns nothing. Dimension prototypes are the 34 verified claim texts grouped
by dimension; there is no hand-written mapping from the synthetic implicit
phrases to labels.

Reproduce the environment and inspect the frozen experiment:

```bash
/Users/tom/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv-embed
.venv-embed/bin/python -m pip install -r evaluation_v1_2/requirements-embedding.txt
.venv-embed/bin/python evaluation_v1_2/scripts/validate_semantic_experiment.py
```

The evaluator refuses to touch the frozen test result again by default. An
explicit reproduction requires
`evaluate_semantic_dimension_router.py --reproduce`.

Model and selection contract:

- FastEmbed `0.8.0` with `BAAI/bge-small-en-v1.5`;
- resolved quantized ONNX repository revision
  `qdrant/bge-small-en-v1.5-onnx-q@52398278842ec682c6f32300af41344b1c0b0bb2`;
- one global threshold, selected on dev dimension exact match;
- ties resolved with the highest threshold (conservative fallback);
- frozen threshold `0.70`; test evaluated once after selection.

The semantic fallback raises overall dimension/evidence exact match from 81.25%
to 93.75% and evidence micro-recall from 84.8% to 96.0%. On the untouched test
split, exact match rises from 27/35 (77.1%) to 34/35 (97.1%), a 20 percentage
point gain. The implicit `v3` group rises from 7/22 to 17/22. All five remaining
errors are the same phrase family—"What should I fill the pot with?"—ranked as
watering instead of soil. That failure is retained; no post-test threshold or
prototype adjustment was made.

FastEmbed uses lightweight ONNX inference rather than requiring PyTorch; see
the [official FastEmbed documentation](https://qdrant.github.io/fastembed/)
and [supported-model table](https://qdrant.github.io/fastembed/examples/Supported_Models/).

## Citation-first answer runtime

`GroundedRAGEngine` completes the evaluated path without requiring a cloud
model or API key. The default generator is extractive by design: each factual
sentence is copied from one `human_verified` claim and followed by its
`claim_id`. Evidence objects include the source URL, section, paragraph, access
date, and review status. Unsupported dimensions expose no claims; the unresolved
Monstera and Hoya common-name mappings request a scientific name.

```bash
python3 -m evaluation_v1_2 ask "How should I water Snake Plant?"
python3 -m evaluation_v1_2 ask --json "Is Aloe Vera pet safe?"
python3 -m evaluation_v1_2 evaluate
python3 -m evaluation_v1_2 verify
```

The end-to-end contract evaluator reuses the frozen route decisions instead of
rerunning or tuning the embedding model. Across all 80 synthetic cases:

- grounded-response validation: 80/80;
- answered citation validity: 100%;
- verified-only evidence: 100%;
- numeric assertions supported by cited evidence: 100%;
- unsafe answers to abstain/clarify cases: 0;
- evidence exact match: 75/80 (93.75%).

On the untouched 35-case test split, behavior is 35/35 and evidence exact match
is 34/35. These remain component/regression results because the questions were
generated from the same evidence being evaluated.

### Optional OpenAI Structured Outputs generator

Natural-language rewriting is opt-in. The adapter uses the Responses API
Structured Outputs helper, requires an explicit model ID, sets `store=False`,
and returns sentence-level evidence IDs. See the
[official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

```bash
python3.12 -m venv .venv-llm
.venv-llm/bin/python -m pip install -r evaluation_v1_2/requirements-llm.txt
export OPENAI_API_KEY="..."
.venv-llm/bin/python -m evaluation_v1_2 ask \
  --generator openai --model "YOUR_MODEL_ID" \
  "How should I water Snake Plant?"
```

Before returning an LLM answer, the runtime verifies that every retrieved
evidence ID is cited exactly once, every citation exists, the rendered answer
contains only structured cited sentences, and no generated number is absent
from its cited evidence. Any API or validation failure returns the extractive
answer and records a sanitized fallback reason. This is deliberately reported
as structural/numeric validation—not proof that every paraphrase is entailed.
