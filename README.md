# PlantCareRAG — dataset-backed v1

The active implementation is now the dataset-backed rebuild in
[`dataset_v1/`](dataset_v1/README.md). It replaces manual knowledge authoring
with a reproducible Kaggle download, license manifest, conflict quarantine,
atomic corpus build, deterministic validation, and a minimal retrieval contract
benchmark. It now also includes a runnable query engine with source-row
citations and structural refusal gates.

Current output:

- 263 accepted plant names;
- 1,315 atomic care entries;
- 124 conflicted names quarantined;
- 650 consistency-filtered context samples;
- test N=658 raw BM25 Hit@1 26.4% / Hit@3 66.9%;
- entity + dimension gate Hit@1 / Hit@3 100% on the template contract test.

Try it:

```bash
python3 dataset_v1/scripts/ask.py "When should I water Snake Plant?"
```

These numbers do not establish botanical truth or real-user QA performance.
The source supports only five shallow care dimensions, and every record remains
labeled as a single-dataset claim. Toxicity, disease, and diagnosis are outside
scope.

The previous hand-built 3-plant experiment remains immutable at Git tag
`v1-frozen`. The abandoned scaling draft remains on branch
`codex/v2-scaling`; neither is mixed into the replacement result.

## v1.2 evidence work

[`evaluation_v1_2/`](evaluation_v1_2/README.md) adds answerability states,
multi-label routing, qualifier-aware refusal, and claim-to-evidence contracts
without changing the frozen v1 metrics. Its first overlay contains 24 care
claims for 10 accepted plants from Extension pages; all remain explicitly
pending human review and are not yet used by the runtime.
