"""Hybrid search: metadata filters + filter planner + BM25 + embeddings, fused by rank."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .data import claim_with_parents, cpc_matches, parse_cpc_query
from .index import Index, normalize_title, tokenize
from .trace import Trace

RRF_K = 60
POOL = 200  # pieces taken from each retriever before fusion
PRE_FILTER_MAX = 0.30  # planner: filter keeps <= 30% of patents -> score only those
OVERFETCH = 4  # post-filter: fetch this many times more than needed, then filter


@dataclass
class Filters:
    cpc: list[str] = field(default_factory=list)  # OR across entries, e.g. ["B60B", "B60C9/20"]
    keywords: list[str] = field(default_factory=list)  # ALL must appear in title or abstract
    title: str | None = None  # exact title (case/punctuation-insensitive)
    before: str | None = None  # YYYY-MM-DD, strictly earlier publication date
    exclude: set[str] = field(default_factory=set)  # doc numbers to drop (e.g. the query patent)
    exclude_pieces: set[str] = field(default_factory=set)  # piece ids to drop (eval: hide a query's own claims)

    def active(self) -> bool:
        return bool(self.cpc or self.keywords or self.title or self.before or self.exclude or self.exclude_pieces)

    def describe(self) -> str:
        parts = []
        if self.cpc:
            parts.append(f"cpc in {self.cpc}")
        if self.keywords:
            parts.append(f"title/abstract has all of {self.keywords}")
        if self.title:
            parts.append(f"title == {self.title!r}")
        if self.before:
            parts.append(f"published before {self.before}")
        if self.exclude:
            parts.append(f"excluding {sorted(self.exclude)}")
        if self.exclude_pieces:
            parts.append(f"hiding {len(self.exclude_pieces)} pieces")
        return "; ".join(parts) or "none"


@dataclass
class Evidence:
    pid: str
    ref: str
    kind: str
    text: str
    bm25_rank: int | None
    dense_rank: int | None


@dataclass
class Hit:
    doc: str
    title: str
    cpc: str
    pub_date: str
    score: float
    evidence: list[Evidence]
    siblings: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    hits: list[Hit]
    strategy: str  # "none" | "pre" | "post"
    n_candidates: int  # patents passing the filter
    timings: dict[str, float]


# --------------------------------------------------------------------------- filters


def patent_mask(idx: Index, f: Filters) -> np.ndarray:
    """Boolean mask over patents. Cheapest/most selective checks first (set lookups)."""
    n = len(idx.patents)
    mask = np.ones(n, dtype=bool)
    if f.title:
        keep = np.zeros(n, dtype=bool)
        keep[idx.title_norm.get(normalize_title(f.title), [])] = True
        mask &= keep
    if f.keywords:
        sets = []
        for kw in f.keywords:
            toks = tokenize(kw)
            if not toks:
                continue
            # a multi-word keyword needs every word present
            s = set.intersection(*(idx.kw_postings.get(t, set()) for t in toks))
            sets.append(s)
        hit = set.intersection(*sets) if sets else set(range(n))
        keep = np.zeros(n, dtype=bool)
        keep[list(hit)] = True
        mask &= keep
    if f.cpc:
        queries = [parse_cpc_query(c) for c in f.cpc]
        keep = np.array([any(cpc_matches(p.cpc_parsed, q) for q in queries) for p in idx.patents])
        mask &= keep
    if f.before:
        mask &= np.array([bool(p.pub_date) and p.pub_date < f.before for p in idx.patents])
    for d in f.exclude:
        if d in idx.by_doc:
            mask[idx.by_doc[d]] = False
    return mask


def plan(idx: Index, f: Filters, pmask: np.ndarray, force: str | None = None) -> str:
    if force in ("pre", "post"):
        return force
    if not f.active():
        return "none"
    return "pre" if pmask.mean() <= PRE_FILTER_MAX else "post"


# --------------------------------------------------------------------------- retrieval


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def _retrieve(idx: Index, qtok, qvec, piece_mask, strategy, need, tr: Trace, mode: str):
    """Return ranked piece indices for BM25 and dense, each already filtered."""
    bm_ranked = dense_ranked = np.array([], dtype=np.int64)
    if mode in ("hybrid", "bm25"):
        with tr.step("BM25 keyword scoring"):
            if strategy == "pre":
                s = idx.bm25.scores(qtok, mask=piece_mask)
                s[~piece_mask] = -1
            else:
                s = idx.bm25.scores(qtok)
            ranked = _top(s, need * (OVERFETCH if strategy == "post" else 1))
            ranked = ranked[s[ranked] > 0]
            if strategy == "post":
                ranked = ranked[piece_mask[ranked]]
            bm_ranked = ranked[:need]
            tr.info(f"{len(bm_ranked)} pieces kept (matched query terms: "
                    f"{sum(t in idx.bm25.postings for t in set(qtok))}/{len(set(qtok))})")
    if mode in ("hybrid", "dense"):
        with tr.step("embedding similarity"):
            if strategy == "pre":
                cand = np.flatnonzero(piece_mask)
                s = idx.emb[cand] @ qvec
                dense_ranked = cand[_top(s, need)]
                tr.info(f"compared against {len(cand)} filtered pieces only (pre-filter)")
            else:
                s = idx.emb @ qvec
                ranked = _top(s, need * (OVERFETCH if strategy == "post" else 1))
                if strategy == "post":
                    before = len(ranked)
                    ranked = ranked[piece_mask[ranked]]
                    tr.info(f"compared against all {len(s)} pieces, then filter kept {len(ranked)}/{before}")
                else:
                    tr.info(f"compared against all {len(s)} pieces")
                dense_ranked = ranked[:need]
    return bm_ranked, dense_ranked


def search(idx: Index, query: str, filters: Filters | None = None, k: int = 10, mode: str = "hybrid",
           force_strategy: str | None = None, trace: Trace | None = None) -> SearchResult:
    tr = trace or Trace()
    f = filters or Filters()
    with tr.step("apply metadata filters"):
        pmask = patent_mask(idx, f)
        piece_mask = pmask[idx.piece_pat]
        for pid in f.exclude_pieces:
            if pid in idx.piece_by_id:
                piece_mask[idx.piece_by_id[pid]] = False
        tr.info(f"filters: {f.describe()}")
        tr.info(f"{int(pmask.sum())}/{len(pmask)} patents, {int(piece_mask.sum())} pieces pass")
    with tr.step("plan filter strategy"):
        strategy = plan(idx, f, pmask, force_strategy)
        why = {"none": "no filters", "pre": f"selective ({pmask.mean():.0%} <= {PRE_FILTER_MAX:.0%}): score only the survivors",
               "post": f"broad ({pmask.mean():.0%} > {PRE_FILTER_MAX:.0%}): score everything, over-fetch x{OVERFETCH}, then filter"}
        tr.info(f"strategy = {strategy}: {why[strategy] if not force_strategy else 'forced by user'}")
    if not pmask.any():
        tr.info("no patent passes the filters -> empty result")
        return SearchResult([], strategy, 0, dict(tr.timings))

    qtok = tokenize(query)
    qvec = None
    if mode in ("hybrid", "dense"):
        with tr.step("embed query"):
            qvec = idx.embed_query(query)
    bm, dn = _retrieve(idx, qtok, qvec, piece_mask, strategy, POOL, tr, mode)
    if strategy == "post" and max(len(bm), len(dn)) < min(POOL, int(piece_mask.sum())) // 2:
        # Over-fetch wasn't enough: too few survivors after filtering. Fall back to pre-filter
        # so a filter can never silently shrink the result list.
        tr.info("post-filter left too few pieces -> falling back to pre-filter")
        strategy = "pre"
        bm, dn = _retrieve(idx, qtok, qvec, piece_mask, strategy, POOL, tr, mode)

    with tr.step(f"fuse rankings ({'reciprocal rank fusion' if mode == 'hybrid' else mode + ' only'})"):
        score: dict[int, float] = {}
        bm_rank = {int(p): r for r, p in enumerate(bm, 1)}
        dn_rank = {int(p): r for r, p in enumerate(dn, 1)}
        for ranks in (bm_rank, dn_rank):
            for p, r in ranks.items():
                score[p] = score.get(p, 0.0) + 1.0 / (RRF_K + r)
        tr.info(f"{len(score)} unique pieces ({len(set(bm_rank) & set(dn_rank))} found by both retrievers)")

    with tr.step("group pieces by patent (best piece wins) + collapse sibling families"):
        by_pat: dict[int, list[int]] = {}
        for p in sorted(score, key=score.get, reverse=True):
            by_pat.setdefault(int(idx.piece_pat[p]), []).append(p)
        hits: list[Hit] = []
        fam_seen: dict[str, Hit] = {}
        for pat_i, plist in sorted(by_pat.items(), key=lambda kv: -score[kv[1][0]]):
            pat = idx.patents[pat_i]
            if pat.family in fam_seen:
                fam_seen[pat.family].siblings.append(pat.doc)
                continue
            ev = [Evidence(idx.pieces[p].pid, idx.pieces[p].ref, idx.pieces[p].kind, idx.pieces[p].text,
                           bm_rank.get(p), dn_rank.get(p)) for p in plist[:3]]
            h = Hit(pat.doc, pat.title, pat.cpc_display, pat.pub_date, score[plist[0]], ev)
            fam_seen[pat.family] = h
            if len(hits) < k:  # past k we keep scanning only to attach siblings to shown hits
                hits.append(h)
        tr.info(f"{len(by_pat)} patents touched -> top {len(hits)} returned")
    return SearchResult(hits, strategy, int(pmask.sum()), dict(tr.timings))


# --------------------------------------------------------------------------- patent as query


def independent_claims(idx: Index, doc: str) -> list[int]:
    p = idx.patents[idx.by_doc[doc]]
    return [c.num for c in p.claims if c.depends_on is None]


def search_by_patent(idx: Index, doc: str, claim: int | None = None, filters: Filters | None = None,
                     k: int = 10, date_cutoff: bool = True, mode: str = "hybrid",
                     trace: Trace | None = None) -> tuple[SearchResult, list[int]]:
    """Prior-art search for an existing patent.

    - no claim given -> search each independent claim, fuse patent rankings (independent claims first:
      dependent claims only add parts to them)
    - dependent claim given -> query = parent chain + own text
    - date cutoff: only patents published strictly before this one (publication-date approximation)
    """
    tr = trace or Trace()
    if doc not in idx.by_doc:
        raise KeyError(f"Patent {doc} not in index")
    pat = idx.patents[idx.by_doc[doc]]
    base = filters or Filters()
    before = base.before
    if date_cutoff and pat.pub_date:
        before = min(before, pat.pub_date) if before else pat.pub_date
    # copy, never mutate the caller's filters
    f = replace(base, exclude=set(base.exclude) | {doc}, before=before)
    claims = [claim] if claim else independent_claims(idx, doc)[:3] or [pat.claims[0].num]
    tr.info(f"query patent {doc} '{pat.title}' published {pat.pub_date}; using claim(s) {claims}")
    results = []
    for c in claims:
        with tr.step(f"search with claim {c}"):
            results.append(search(idx, claim_with_parents(pat, c), f, k=k * 2, mode=mode, trace=tr))
    if len(results) == 1:
        r = results[0]
        r.hits = r.hits[:k]
        return r, claims
    with tr.step("fuse results across independent claims"):
        agg: dict[str, float] = {}
        best: dict[str, Hit] = {}
        for r in results:
            for rank, h in enumerate(r.hits, 1):
                agg[h.doc] = agg.get(h.doc, 0.0) + 1.0 / (RRF_K + rank)
                if h.doc not in best or h.score > best[h.doc].score:
                    best[h.doc] = h
        hits = [best[d] for d in sorted(agg, key=agg.get, reverse=True)][:k]
        for h in hits:
            h.score = agg[h.doc]
    return SearchResult(hits, results[0].strategy, results[0].n_candidates, dict(tr.timings)), claims
