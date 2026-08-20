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
without changing the frozen v1 metrics. Its evidence overlay started with 24
care claims for 10 accepted plants and expanded to 38 through human-review
atomization and coverage findings. Only individually verified claims are
visible through the runtime evidence boundary.

The citation-first v1.3 runtime completes the local RAG path over that verified
overlay: entity linking → dimension routing → structured evidence retrieval →
extractive grounded answer → inline claim/source citations. It defaults to a
dependency-free lexical router; the frozen BGE semantic fallback is optional.

```bash
# Ask with the dependency-free router
python3 -m evaluation_v1_2 ask "How should I water Snake Plant?"

# Rebuild the 80-case grounded-answer contract artifact
python3 -m evaluation_v1_2 evaluate

# Run every v1.2/v1.3 offline validator and unit test
make verify
```

For implicit wording, create a Python 3.10+ environment, install the pinned
embedding dependency, and enable the frozen semantic fallback:

```bash
python3.12 -m venv .venv-embed
.venv-embed/bin/python -m pip install -r evaluation_v1_2/requirements-embedding.txt
.venv-embed/bin/python -m evaluation_v1_2 ask --semantic \
  "What should I know before watering my Aloe Vera again?"
```

The 80 synthetic questions are evidence-conditioned regression cases, not a
real-user benchmark. On the frozen semantic routes, all answers pass citation
and verified-evidence checks; test behavior is 35/35 and test evidence exact
match is 34/35. The remaining retrieval error is preserved rather than tuned
after test access.
