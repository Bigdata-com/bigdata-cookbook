# Entity-Sentence Rerank Experiment

Standalone experiment to compare chunk-level retrieval signals vs entity-local
sentence extraction + embedding rerank + sentence-level labeling.

Does **not** modify the main thematic screener pipeline.

## Hypothesis

Bigdata chunk `relevance` and `sentiment` are weak proxies. After retrieval:

1. Use **detection coordinates** (`start`/`end`) to extract the sentence containing each
   entity mention that appears in the **search plan basket** (query universe).
2. Rerank with cosine similarity between the extracted sentence and the plan's `search_query`
   using OpenAI `text-embedding-3-small`.
3. Label the extracted sentence (experiment) vs the full chunk (baseline).

Each entity mention yields one **entity–sentence–chunk** record.

## Prerequisites

- `uv sync` in `Thematic_Screener_CLI/`
- `OPENAI_API_KEY` and Bigdata credentials in `.env`
- An existing run with plans, e.g. `runs/run_20260629_081049/`
- Dev dependency for plots: `uv add --dev matplotlib`

## Run

```bash
cd Thematic_Screener_CLI

# Scores + extraction only (no LLM labeling cost)
uv run python experiments/entity_sentence_rerank/run_experiment.py \
  --source-run runs/run_20260629_081049 \
  --plans "Quantum_hardware_developers.json,Cryogenics_and_specialized_components.json" \
  --chunk-percentage 0.002 \
  --sample-size 100 \
  --skip-labeling

# Full comparison (100 chunks → N entity-sentence records; 2× labeling passes)
uv run python experiments/entity_sentence_rerank/run_experiment.py \
  --source-run runs/run_20260629_081049 \
  --sample-size 100
```

Re-use cached retrieval:

```bash
uv run python experiments/entity_sentence_rerank/run_experiment.py --skip-retrieval
```

## Outputs (`experiments/entity_sentence_rerank/outputs/`)

| File | Description |
|------|-------------|
| `retrieved_chunks.jsonl` | Sampled chunks with plan/query metadata |
| `entity_sentence_records.jsonl` | Entity–sentence–chunk records after extraction |
| `scored_records.csv` | Records with embedding + baseline scores |
| `labeled_comparison.csv` | Baseline vs experiment labels (if labeling enabled) |
| `review_queue.csv` | Manual review slice (label disagreements / rank gaps) |
| `summary.json` / `summary.md` | Aggregate metrics |
| `relevance_scatter.png` | Search vs embedding relevance |

## Extraction rules

1. Filter `detections` where `type == "entity"` and `id` is in the plan basket (`plan_entity_ids`).
2. For each matching detection, find sentence boundaries around `[start:end)` using coordinate walk + regex guards.
3. Emit one record per detection (multiple entities → multiple records from the same chunk).
4. If no query-entity detections exist, emit one fallback record with `extraction_ok=false`.

## Golden set (GPT-5.5, provenance-locked)

Build a gold standard where the label is fixed to the retrieval pathway
(`leaf_label` if relevant, otherwise `unclear`):

```bash
# 100-chunk set (2 plans)
uv run python experiments/entity_sentence_rerank/build_golden_set.py \
  --chunks-path experiments/entity_sentence_rerank/outputs/retrieved_chunks.jsonl \
  --output-dir experiments/entity_sentence_rerank/outputs \
  --limit 100

# 500-chunk set (all 7 quantum plans)
uv run python experiments/entity_sentence_rerank/run_experiment.py \
  --all-plans --sample-size 500 \
  --output-dir experiments/entity_sentence_rerank/outputs_500 \
  --skip-labeling

uv run python experiments/entity_sentence_rerank/build_golden_set.py \
  --chunks-path experiments/entity_sentence_rerank/outputs_500/retrieved_chunks.jsonl \
  --output-dir experiments/entity_sentence_rerank/outputs_500 \
  --limit 500

uv run python experiments/entity_sentence_rerank/run_experiment.py \
  --all-plans --sample-size 500 \
  --output-dir experiments/entity_sentence_rerank/outputs_500 \
  --skip-retrieval --provenance-locked

uv run python experiments/entity_sentence_rerank/run_assess.py \
  --golden-path experiments/entity_sentence_rerank/outputs_500/golden_set.csv \
  --records-path experiments/entity_sentence_rerank/outputs_500/labeled_comparison.csv \
  --output-dir experiments/entity_sentence_rerank/outputs_500
```

Outputs:
- `outputs/golden_set.csv`
- `outputs/golden_set.jsonl`
- `outputs/golden_set_manifest.json`

Columns include `gold_relevant`, `gold_label`, `gold_materiality`,
`gold_evidence_quality`, and `gold_motivation`.

## Assessment against golden set

```bash
uv run python experiments/entity_sentence_rerank/run_assess.py
```

Outputs:
- `outputs/assessment.json` / `assessment.md`
- `outputs/assessment_merged.csv` (chunk-level merge of scores + gold)

Metrics include average precision and P/R@K for search, embedding, and evidence
scores, plus labeling accuracy when `labeled_comparison.csv` exists.

The assessment also writes `embed_threshold_grid.csv`, sweeping
`sentence_label + embedding_relevance >= t` against the golden set.
Use `--embed-threshold 0.52` to highlight a chosen operating point.

## Provenance-locked labeling (optional)

`label_provenance.py` mirrors the golden-set prompt for fair baseline vs sentence
comparison. Wire it into `run_experiment.py` when re-running labeling passes.

## Cost / latency

- Fresh retrieval (2 plans, 0.2%): ~1–2 min Bigdata API
- Embeddings (~200 records): pennies
- Labeling (2 passes × N records): ~$0.10–0.30 depending on record count
