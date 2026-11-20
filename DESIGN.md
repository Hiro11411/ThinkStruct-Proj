# DESIGN: running the Part 1 search engine on 10 million patents

Audience: an engineer starting on this tomorrow. Goal: the same search as Part 1 (hybrid keyword +
embedding search over claims, abstracts and paragraphs, with CPC / keyword / title / date filters),
over ~10M patents, with weekly updates. The design is deliberately simple: few components, all
off-the-shelf. Its weak points are listed at the end.

## 1. Sizing (from the measured sample, scaled up)

| Quantity | Sample (measured) | 10M patents | How |
|---|---|---|---|
| Pieces per patent (abstract + claims + paragraphs) | 50.4 (32,252 / 640) | **~504M pieces** | ×10M |
| Text per patent | 24 KB | ~240 GB text | raw XML in object storage ~1 TB |
| Vectors, 384-dim float32 | 49.5 MB | **~774 GB** | 1,536 B each |
| float16 / int8 / binary | | 387 / 194 / **24 GB** | 768 / 384 / 48 B each |
| HNSW graph links (M=16) | | ~64 GB | ~128 B per vector |
| Embedding speed, laptop CPU | 26 pieces/s | 225 days on one CPU | → GPUs |
| Query embedding (CPU) | 33 ms | 33 ms | does not grow with corpus |
| Search excluding embedding | 3–14 ms (brute force) | target 50–100 ms (ANN) | |
| New documents per week | 10 per file (sample) | ~15k (USPTO grants + applications) | ~750k pieces/week |

**Biggest number:** ~500M vectors. Full-precision vectors don't fit in RAM on a reasonable cluster,
so the index keeps **binary vectors in RAM** (24 GB) and **float16 copies on disk**. It searches on binary,
fetches 4x more candidates than needed, and rescores them with float16. Published results:
binary + rescore keeps ~96% of full-precision retrieval quality
([HF embedding quantization](https://huggingface.co/blog/embedding-quantization)). Re-measure
this on our own eval set before relying on it.

## 2. Components

```
                 USPTO bulk XML (weekly)                         users / API clients
                          |                                              |
                          v                                              v
  +-----------+    +-------------+    +-------------+         +--------------------+
  |  Fetcher  |--->| Object store|--->|   Parser    |         |  Search API        |
  | (cron)    |    | raw/ parsed/|    |  workers    |         |  (stateless, x2+)  |
  +-----------+    +-------------+    +------+------+         | - embed query (CPU)|
        |                                    |                | - filter planner   |
        |          +-------------------------+                | - fuse BM25 + kNN  |
        v          v                                          | - group by patent  |
  +-----------------------------+     +-------------+         +---------+----------+
  | Postgres                    |<----| Embedding   |                   |
  | - documents (metadata)      |     | workers GPU |                   v
  | - doc_status (tracking)     |     +------+------+         +--------------------+
  | - jobs (queue, SKIP LOCKED) |            |                | OpenSearch cluster |
  +-----------------------------+            +--------------->| - BM25 text index  |
                                                              | - kNN index        |
                                                              |   (binary+rescore) |
                                                              | - filter fields    |
                                                              +--------------------+
```

| Component | Job | Choice and why |
|---|---|---|
| Fetcher | Downloads the weekly USPTO bulk files (application + grant XML) and the one-off backfill | Cron job + Python. Writes `raw/<week>/<doc>.xml` to object storage. |
| Object store | Source of truth for raw and parsed documents | S3 (or GCS). Cheap and durable; any later stage can be replayed from it. |
| Parser workers | XML → clean JSON: title, abstract, claims (numbered, dependency, canceled dropped), paragraphs with numbers, CPC, **filing + publication + priority dates**, family id | The Part 1 cleaning code (`patsearch/data.py`) plus validation (section 5). CPU, horizontally scaled. |
| Chunker | Clean JSON → pieces (abstract / claim / paragraph, split over 350 words) | Same code as Part 1 (`make_pieces`). Runs inside the parser worker. |
| Embedding workers | Pieces → 384-dim vectors (bge-small) | GPU workers pulling batches from the queue. Model name and version stored with every vector. |
| Job queue | Hands out work, retries, dead-letter | A Postgres `jobs` table with `SELECT ... FOR UPDATE SKIP LOCKED`. One less system than Kafka/SQS, and fine at ~15k docs/week. |
| Postgres | Patent metadata, per-document status, the queue | Single primary + read replica. 10M metadata rows is small for Postgres. |
| OpenSearch | BM25 + vector search + filters in **one** engine | One engine does keywords, kNN (HNSW with binary quantization and rescoring) and filtered search on keyword fields (cpc, dates). Hybrid queries fuse BM25 and kNN by rank. Fewer moving parts than BM25 in one system and vectors in another. |
| Search API | The Part 1 query path | Stateless Python service: embed query → planner → one hybrid OpenSearch query → group by patent → (Part 3 rerank on top 30). Scales by adding replicas. |
| Monitoring | Is it healthy and complete? | Status counts from Postgres, API latency/error metrics, nightly count reconciliation (section 6). |

### Data model (Postgres)

```sql
documents(doc_number PK, title, cpc_subclass, cpc_group, cpc_subgroup,
          filing_date, publication_date, priority_date, family_id,
          n_claims, n_paragraphs, content_hash, model_version, updated_at)

doc_status(doc_number PK, stage TEXT,        -- fetched | parsed | embedded | indexed
           status TEXT,                      -- ok | failed | quarantined
           attempts INT, last_error TEXT, updated_at)

jobs(id PK, kind TEXT,                       -- parse | embed | index
     doc_number, payload JSONB, state TEXT,  -- queued | running | done | dead
     attempts INT, run_after TIMESTAMPTZ, locked_by, locked_at)
```

### Search index (OpenSearch), one document per piece

`piece_id` (`<doc>:c3`, `<doc>:p12`), `doc_number`, `kind`, `text` (BM25), `vector` (knn_vector,
384, binary-quantized, on-disk rescore), `cpc_subclass`, `cpc_group`, `cpc_full`,
`publication_date`, `filing_date`, `family_id`, `title_keyword`.

CPC is stored **as separate components**, not one string. Prefix-matching the raw code is a bug
we found in the sample: "B60B1" would also match B60B 11, 15, 19.

## 3. Pipelines

**A. Bulk backfill (one-off, ~10M docs)**
1. Fetcher lists every historical weekly file and enqueues one `parse` job per document.
2. Parser workers (~50 spot CPU workers) parse, validate, chunk, write `parsed/<doc>.json`, upsert
   `documents`, and enqueue `embed`.
3. Embedding workers (8 spot GPUs) embed in batches of ~256 pieces and write vectors to
   `vectors/<doc>.npy` (float16).
4. Indexer bulk-loads into a **new** OpenSearch index (`patents_v1`). When the doc counts match
   Postgres, the alias `patents` is pointed at it.

Throughput: bge-small on one A10G-class GPU should do roughly 3–5k short pieces/s (to be measured).
That makes 504M pieces ≈ 30–45 GPU-hours, so 8 spot GPUs finish in about 5 hours.

**B. Weekly update (~15k docs)**
Same stages for that week's file only, writing into the live index. About 750k pieces is a few GPU-minutes.
It's idempotent (section 5), so re-running a week is always safe.

**C. Re-embed (model change)**
Embed everything with the new model into `patents_v2` alongside the old index, run the eval set on
both, then swap the alias. Never mix model versions in one index: vectors from different models
can't be compared.

**D. Query**
1. API embeds the query (CPU, ~33 ms measured).
2. The filter planner estimates how many documents pass the filters, using a Postgres count or
   OpenSearch cardinality.
3. Selective filters (< ~1% of docs) → exact scoring of the filtered subset.
   Broad filters → ANN with filtering and over-fetch.
   This is the same rule as Part 1, where the sample shows it: for the broad B60B filter,
   post-filter 6.0 ms vs pre-filter 13.7 ms.
4. Hybrid BM25 + kNN query → fuse by rank → group by patent, collapse families → top 30 →
   (Part 3 coverage rerank, fixed cost because it only touches 30 documents) → return.

## 4. Cost (rough, on-demand list prices; check before budgeting)

| Item | Sizing | ~ $/month |
|---|---|---|
| OpenSearch | 3 data nodes × 128 GB RAM (binary vectors + HNSW + BM25 caches, 1 replica), ~1.5 TB SSD | ~3,000 |
| Postgres | 1 primary (2 vCPU / 16 GB) + 500 GB storage | ~250 |
| Object storage | ~1 TB raw + parsed + float16 vectors (~0.4 TB) | ~35 |
| Search API | 2 small CPU instances | ~200 |
| Weekly pipeline | a few GPU-minutes + spot CPU parsers | ~50 |
| **Total running** | | **~3,500 / month** |
| One-off backfill | ~40 GPU-hours + ~500 CPU-hours (spot) | **~100–300 once** |

Cost levers, in order:
- self-host OpenSearch or Qdrant on reserved instances (often 2–3x cheaper);
- drop float16 rescoring to on-demand reads from object storage;
- index only claims + abstracts for ANN (~17 pieces per patent instead of 50) and fetch paragraphs
  by `doc_number` only for the top 30 in Part 3. That cuts the vector count about 3x, at some cost to recall.

## 5. Error handling

- **Idempotent by key.** Every write is keyed by `doc_number` (+ `piece_id`) with a
  `content_hash`. Re-running any job overwrites with the same result; nothing is duplicated. If the
  hash hasn't changed, the job is skipped.
- **Retries.** A failed job goes back to `queued` with `run_after = now() + 2^attempts minutes`.
  After 5 attempts it becomes `dead` and `doc_status.status = failed` with `last_error`.
- **Dead-letter replay.** `replay --stage embed --since 2026-11-01` re-enqueues dead jobs once
  the bug is fixed.
- **Validation before indexing (from problems found in the sample).** In 640 sample patents, claim 1
  lost its opening words in 597, 119 had empty descriptions, and 73 claims were cut off at "(".
  The parser checks: claim count > 0, claim numbers increasing, canceled ranges consumed, cut-off
  claims flagged, description non-empty. A document that fails a hard check is `quarantined`:
  kept out of the index, listed for review. Soft problems (empty description) are indexed with a
  flag, so search still works on claims + abstract.
- **Crash safety.** A worker that dies leaves `locked_at` stale. A sweeper re-queues jobs
  locked for more than 15 minutes.
- **Index safety.** Bulk loads go into a new index and only replace the live one by alias swap,
  after the counts check. A bad backfill never reaches users.

## 6. Tracking: what's in the system and is it healthy

- `doc_status` answers "where is document X?" and "how many are stuck?":
  `SELECT stage, status, count(*) FROM doc_status GROUP BY 1, 2;`
- Weekly batch report: fetched / parsed / embedded / indexed / failed / quarantined counts.
  **Alert if failed + quarantined is more than 1% of the batch** (a parser regression usually shows up here first).
- Nightly reconciliation: Postgres `indexed` count = OpenSearch distinct `doc_number` count.
  A mismatch triggers a re-index of the difference.
- Quality: the Part 1 eval set (`python -m patsearch eval`) runs against every new index before
  the alias swap. A drop in hit@10 blocks the swap.
- Query metrics: p50/p95 latency per stage (embed, search, rerank), errors, empty-result rate.

## 7. Major challenges at scale (and what we do about each)

1. **500M vectors.** Memory is the cost driver. Binary in RAM + float16 rescore on disk;
   option to index only claims + abstracts (section 4).
2. **Filtered vector search.** Plain "ANN then filter" returns too few results for selective filters
   (the over-filtering problem; e.g. pgvector before 0.8, https://www.postgresql.org/about/news/pgvector-0.8.0-released-2952/).
   The planner switches to exact scoring of the filtered set for selective filters, and over-fetches for broad ones.
3. **Hybrid fusion across shards.** BM25 scores aren't comparable across shards or queries, so we
   fuse by rank (RRF), not by raw score.
4. **Parser quality.** It's the most likely source of silent errors (section 5). It's measured, not assumed.
5. **Dates.** The sample only has publication dates. Prior art depends on filing/priority dates,
   so the parser must extract them from the XML. Without them the date filter is only an approximation.
6. **Duplicates and families.** One invention appears as an application, a grant and continuations.
   Use the source's family ids where available; fall back to MinHash on abstracts (the sample uses
   exact Jaccard, which is O(n²) and won't scale).
7. **Model upgrades** mean re-embedding 500M pieces. Blue/green indexes (pipeline C) make that
   safe; it costs ~$100–300 each time.
8. **Hot spots.** B60B is half the sample. Real CPC classes are just as skewed, so shard by
   `doc_number` hash, not by CPC.

## 8. What I would build first

1. Postgres schema + job queue + parser/chunker workers (reusing `patsearch/data.py`).
2. OpenSearch index mapping + bulk loader, tested on the 640-patent sample.
3. Search API porting `patsearch/search.py` to one OpenSearch hybrid query.
4. Backfill one year (~1M docs) and measure real throughput and latency before buying the full cluster.

The proof of concept for this part is in `poc/`.
