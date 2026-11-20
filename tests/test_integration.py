"""End-to-end checks on the real index. Skipped until `python -m patsearch build` has run."""
import os

import pytest

from patsearch import index as index_mod
from patsearch.data import cpc_matches, parse_cpc_query
from patsearch.search import Filters, search, search_by_patent

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(index_mod.INDEX_DIR, "meta.pkl")),
                                reason="index not built")


@pytest.fixture(scope="module")
def idx():
    return index_mod.load()


def test_counts(idx):
    assert len(idx.patents) == 640
    assert idx.emb.shape[0] == len(idx.pieces)


def test_claim_finds_its_own_patent(idx):
    """Sanity: with self allowed, a claim's own patent should rank in the top 3 for most patents."""
    pats = [p for p in idx.patents if p.paragraphs][:40]
    found = sum(p.doc in [h.doc for h in search(idx, p.claims[0].text, k=3).hits] for p in pats)
    assert found / len(pats) >= 0.8


def test_every_hit_respects_filters(idx):
    f = Filters(cpc=["B60C"], before="2024-10-01")
    res = search(idx, "pneumatic tire tread with circumferential grooves", f, k=20)
    q = parse_cpc_query("B60C")
    for h in res.hits:
        p = idx.patents[idx.by_doc[h.doc]]
        assert cpc_matches(p.cpc_parsed, q) and p.pub_date < "2024-10-01"


def test_patent_search_only_returns_earlier_filings(idx):
    pat = idx.patents[-1]
    res, _ = search_by_patent(idx, pat.doc)
    assert res.hits and all(h.pub_date < pat.pub_date for h in res.hits)
    assert pat.doc not in [h.doc for h in res.hits]
