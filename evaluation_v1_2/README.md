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
- a 10-plant, 24-claim authoritative-source overlay awaiting human review.

It does **not** add embeddings or LLM generation. Those components should be
evaluated only after a frozen real-query set and reviewed evidence overlay exist.

## Run it

~~~bash
python3 evaluation_v1_2/scripts/validate_contract.py
python3 evaluation_v1_2/scripts/validate_evidence_overlay.py
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

The initial overlay covers 10 accepted entities with 24 care claims from 10 NC
State Extension Plant Toolbox pages. Coverage is intentionally uneven: a claim
is included only when the plant page states it directly, so fertilizer is not
manufactured for plants whose page only documents light and water.

These records are marked
`agent_source_checked_pending_human_review`, not `human_verified`, and are not
yet used by the runtime. The manual pass is tracked in
[`REVIEW_CHECKLIST.md`](REVIEW_CHECKLIST.md). Genus-level entities and common-
name-to-species mappings carry explicit scope warnings.

## Next data milestone

Before adding semantic retrieval or generation:

1. Complete human review of the 24-claim overlay.
2. Collect and manually label 50–80 real English questions.
3. Freeze query IDs, hashes, evidence IDs, and the dev/test split.
4. Report structured retrieval and answerability baselines.
