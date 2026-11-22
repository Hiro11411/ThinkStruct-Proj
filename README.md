# Patent prior-art search (Thinkstruct take-home)

**Problem:** given a patent claim, find earlier filings that disclose its parts.
Part 1 (this MVP) is a hybrid search engine: BM25 keywords + embeddings (bge-small), fused by
reciprocal rank, with classification / keyword / title / date filters and a filter planner.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (cmd / PowerShell)
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt   # CPU-only torch; no GPU or API key needed (~6 min on Windows)
```

Put the provided `patents_ipa*.json` files in `data/raw/` (not committed), or set `PATSEARCH_DATA`.

**`build` takes about 20 minutes on a laptop CPU.** It embeds ~32,000 pieces at roughly 26 pieces/s
and downloads the embedding model (~130 MB) once. It only needs to run once; every other command loads
the saved index.

## Run

```bash
python -m patsearch build                     # one-off: clean data, build BM25 + embeddings -> index/
python -m patsearch search "tire tread with sipes" --cpc B60C --kw groove --before 2024-09-01
python -m patsearch patent 20240051333        # prior art for a patent (independent claims, date cutoff)
python -m patsearch patent 20240051333 --claim 3
python -m patsearch show 20240051333          # cleaned claims, dependency, broken/cut-off flags
python -m patsearch bench                     # timing table: no / narrow / broad filter, pre vs post
python -m patsearch eval --show-misses        # results table: keyword / embeddings / merged (tune pile)
python -m patsearch eval --pile report        # final reported numbers (don't tune on these)
```

Every command prints a numbered trace of each step (what it did, counts, milliseconds) to stderr.
Add `--quiet` before the command to hide it, e.g. `python -m patsearch --quiet search "..."`.

Filters: `--cpc` (repeatable, OR; `B60B`, `B60B27`, `B60C11/03`), `--kw` (repeatable, all must appear
in title or abstract), `--title` (exact), `--before YYYY-MM-DD`, `-k`, `--mode hybrid|bm25|dense`,
`--strategy pre|post` (override the planner).

## Tests

```bash
python -m pytest -q          # unit tests (no model) + integration tests (once the index is built)
```

## Missing-field rule

Every patent is kept. 119 of 640 patents have a description made only of empty strings; they are
searched on abstract + claims. Claim 1 lost its number and opening words in 597 patents; it is rebuilt
as "A <title>, comprising: ...". Canceled claims are dropped; claim numbers are read from the text.

## Known approximations

- Dates come from the file names and are publication dates, not filing dates; the date cutoff is
  "published strictly before". The brief describes the file date as the week the applications were
  filed, but every file date is a Thursday and matches the publication number in each record
  (`US20240051333A1-20240215.XML`), which is the USPTO publication schedule.
- The source drops the slash in CPC codes (`B60B104FI`); we assume a 2-digit subgroup (`B60B 1/04`).
