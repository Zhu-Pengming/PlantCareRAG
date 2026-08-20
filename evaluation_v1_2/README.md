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

## Benchmark boundary

The query annotation seed contains authored contract examples, not observed
user questions. Every record is marked as authored and benchmark-ineligible.
The future real-query set must retain an immutable source URL and raw wording,
receive manual answerability and gold-evidence labels, and be split by the
stored unsalted SHA-256 of its ID.

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

## Next data milestone

The evidence review milestone is complete. Before adding semantic retrieval or
generation:

1. Collect and manually label 50–80 real English questions.
2. Freeze query IDs, hashes, evidence IDs, and the dev/test split.
3. Report structured retrieval and answerability baselines.
