"""Test harness: one command prints the results table.

Two automatic test sets (no hand labels needed):

1. own-description: query = a patent's claim 1. Its own claims and abstract are hidden, so the only
   way to find the patent is through its description paragraphs. Correct answer = that patent.
   Optimistic: same author, same reference numerals, same words. Good for comparing setups,
   not an absolute accuracy number.
2. siblings: query = claim 1 of a patent in a sibling family (near-identical abstracts, same
   applicant). The patent itself is hidden. Correct answer = any other family member. A small
   cross-patent answer key. Date cutoff off, since siblings publish the same week.

Each set is split into a tuning pile and a report pile (fixed seed) so later tuning
can't overfit the number we report.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

from .index import Index
from .search import Filters, search
from .trace import Trace

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Case:
    name: str
    query: str
    filters: Filters
    relevant: set[str]  # doc numbers that count as correct


def own_description_cases(idx: Index, n: int = 50, seed: int = 0) -> list[Case]:
    pats = [p for p in idx.patents if p.paragraphs and p.claims]
    pats = random.Random(seed).sample(pats, min(n, len(pats)))
    cases = []
    for p in pats:
        hidden = {pc.pid for pc in idx.pieces if pc.doc == p.doc and pc.kind != "para"}
        cases.append(Case(f"own:{p.doc}", p.claims[0].text, Filters(exclude_pieces=hidden), {p.doc}))
    return cases


def sibling_cases(idx: Index) -> list[Case]:
    fams: dict[str, list] = {}
    for p in idx.patents:
        fams.setdefault(p.family, []).append(p)
    cases = []
    for members in fams.values():
        if len(members) < 2:
            continue
        for p in members:
            if p.claims:
                others = {m.doc for m in members if m.doc != p.doc}
                cases.append(Case(f"sib:{p.doc}", p.claims[0].text, Filters(exclude={p.doc}), others))
    return cases


def split(cases: list[Case], seed: int = 1) -> tuple[list[Case], list[Case]]:
    cs = cases[:]
    random.Random(seed).shuffle(cs)
    half = len(cs) // 2
    return cs[:half], cs[half:]


def first_correct_rank(hits, relevant: set[str]) -> int | None:
    for r, h in enumerate(hits, 1):
        if h.doc in relevant or relevant & set(h.siblings):
            return r
    return None


def run(idx: Index, cases: list[Case], mode: str, k: int = 10, trace: Trace | None = None) -> dict:
    tr = trace or Trace()
    ranks, times = [], []
    for c in cases:
        t0 = time.perf_counter()
        res = search(idx, c.query, c.filters, k=k, mode=mode)
        times.append((time.perf_counter() - t0) * 1000)
        ranks.append(first_correct_rank(res.hits, c.relevant))
    found = [r for r in ranks if r]
    out = {
        "n": len(cases),
        "hit@1": sum(r == 1 for r in found) / len(cases),
        "hit@10": len(found) / len(cases),
        "mrr": sum(1 / r for r in found) / len(cases),
        "avg_rank": statistics.mean(found) if found else float("nan"),
        "ms": statistics.median(times),
        "misses": [c.name for c, r in zip(cases, ranks) if r is None],
    }
    tr.info(f"{mode:6s}: hit@10 {out['hit@10']:.0%}, MRR {out['mrr']:.3f}, {len(out['misses'])} misses, "
            f"median {out['ms']:.0f} ms/query")
    return out
