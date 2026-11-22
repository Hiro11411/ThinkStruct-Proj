"""Search, filter and planner rules on a tiny hand-made index (fake embeddings, no model download)."""
import zlib

import numpy as np
import pytest

from patsearch.data import Claim, Patent, make_pieces, parse_cpc
from patsearch.index import BM25, Index, tokenize
from patsearch.search import Filters, patent_mask, plan, search, search_by_patent
from patsearch.trace import Trace


def _pat(doc, title, abstract, cpc, date, claim):
    return Patent(doc, title, abstract, cpc, parse_cpc(cpc), date, [(1, abstract + " described in detail.")],
                  [Claim(1, claim, None, False, False)], family=doc)


@pytest.fixture
def idx():
    pats = [
        _pat("A1", "SPOKE", "A bicycle spoke with an axle body.", "B60B104FI", "2024-01-04", "A spoke with an axle body."),
        _pat("A2", "WHEEL RIM", "A steel wheel rim for trucks.", "B60B2100FI", "2024-02-01", "A rim made of steel."),
        _pat("A3", "TIRE", "A pneumatic tire with tread grooves and sipes.", "B60C1103FI", "2024-03-07", "A tire with sipes."),
        _pat("A4", "TIRE", "A tire with a carcass ply and bead core.", "B60C900FI", "2024-04-04", "A tire with a bead core."),
        _pat("A5", "HITCH", "A trailer hitch with a ball mount.", "B60D106FI", "2024-05-02", "A hitch with a ball mount."),
    ]
    pieces = make_pieces(pats)
    bm25 = BM25([tokenize(p.text) for p in pieces])
    # fake embedding: bag of words hashed to 64 dims -> behaves like a crude semantic model
    def vec(text):
        v = np.zeros(64, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % 64] += 1  # fixed hash: built-in hash() is randomized per run
        return v / (np.linalg.norm(v) or 1)
    emb = np.stack([vec(p.text) for p in pieces])
    ix = Index(pats, pieces, bm25, emb, "fake")
    ix.embed_query = lambda text, prefix=True: vec(text)
    ix._model = "fake-model"  # pretend the model is already loaded (warm process)
    return ix


def docs(res):
    return [h.doc for h in res.hits]


def test_no_filter_finds_obvious_match(idx):
    assert docs(search(idx, "tire sipes tread grooves"))[0] == "A3"


def test_cpc_filter_returns_only_matching(idx):
    res = search(idx, "tire wheel spoke hitch", Filters(cpc=["B60C"]))
    assert docs(res) and set(docs(res)) <= {"A3", "A4"}


def test_deep_cpc_filter(idx):
    assert docs(search(idx, "tire", Filters(cpc=["B60C11/03"]))) == ["A3"]


def test_keyword_filter_on_title_or_abstract(idx):
    res = search(idx, "tire", Filters(keywords=["bead"]))
    assert docs(res) == ["A4"]


def test_exact_title_filter_case_insensitive(idx):
    assert set(docs(search(idx, "anything about a wheel", Filters(title="wheel rim")))) == {"A2"}


def test_date_cutoff_is_strict(idx):
    res = search(idx, "tire spoke rim hitch", Filters(before="2024-03-07"))
    assert set(docs(res)) <= {"A1", "A2"}


def test_impossible_filter_returns_empty(idx):
    res = search(idx, "tire", Filters(cpc=["B60C"], keywords=["hitch"]))
    assert res.hits == [] and res.n_candidates == 0


def test_planner_picks_pre_for_selective_and_post_for_broad(idx):
    narrow = Filters(cpc=["B60D"])  # 1/5 = 20%
    broad = Filters(cpc=["B60B", "B60C"])  # 4/5 = 80%
    assert plan(idx, narrow, patent_mask(idx, narrow)) == "pre"
    assert plan(idx, broad, patent_mask(idx, broad)) == "post"
    assert plan(idx, Filters(), patent_mask(idx, Filters())) == "none"


def test_pre_and_post_filter_agree(idx):
    f = Filters(cpc=["B60C"])
    a = search(idx, "tire with sipes", f, force_strategy="pre")
    b = search(idx, "tire with sipes", f, force_strategy="post")
    assert docs(a) == docs(b)


def test_patent_query_excludes_itself_and_later_filings(idx):
    res, claims = search_by_patent(idx, "A4")
    assert claims == [1]
    assert "A4" not in docs(res) and "A5" not in docs(res)


def test_patent_query_does_not_mutate_callers_filters(idx):
    f = Filters(cpc=["B60C"])
    search_by_patent(idx, "A4", filters=f)
    assert f.before is None and f.exclude == set()


def test_exclude_pieces_hides_only_those_pieces(idx):
    """Eval relies on this: hide A3's claim + abstract, A3 must still be findable via its paragraph."""
    hidden = {"A3:abs", "A3:c1"}
    res = search(idx, "tire sipes", Filters(exclude_pieces=hidden))
    assert res.hits[0].doc == "A3"
    assert all(e.pid not in hidden for h in res.hits for e in h.evidence)


def test_eval_harness_runs(idx):
    from patsearch import evaluate as ev
    cases = ev.own_description_cases(idx, n=5)
    assert len(cases) == 5 and all(c.filters.exclude_pieces for c in cases)
    for mode in ev.MODES:
        r = ev.run(idx, cases, mode)
        assert 0 <= r["hit@10"] <= 1 and r["n"] == 5


def test_modes_run(idx):
    for mode in ("bm25", "dense", "hybrid"):
        assert search(idx, "steel rim", mode=mode).hits


def test_trace_records_every_step(idx):
    tr = Trace()
    search(idx, "tire", Filters(cpc=["B60C"]), trace=tr)
    for step in ("apply metadata filters", "plan filter strategy", "BM25 keyword scoring", "embedding similarity"):
        assert step in tr.timings


# ---------------------------------------------------------------- Batch A: trace timings
def test_model_load_is_timed_separately_from_query_embedding(idx):
    """A1: the cold-start model load must not be counted as 'embed query'."""
    load_calls = []

    def fake_model_loader():
        load_calls.append(1)
        idx._model = "loaded"
        return idx._model

    idx._model = None
    idx.model = fake_model_loader
    first_trace = Trace()
    search(idx, "tire sipes", trace=first_trace)
    assert "load model (first query in this process only)" in first_trace.timings
    assert "embed query" in first_trace.timings

    second_trace = Trace()
    search(idx, "tire sipes", trace=second_trace)
    assert "load model (first query in this process only)" not in second_trace.timings
    assert len(load_calls) == 1


def test_multi_claim_timings_do_not_overwrite_each_other(idx):
    """A2: each independent claim's inner steps get their own timing key."""
    patent = idx.patents[idx.by_doc["A4"]]
    patent.claims.append(Claim(2, "A tire comprising a tread with sipes.", None, False, False))
    trace = Trace()
    result, claims_used = search_by_patent(idx, "A4", trace=trace)
    assert claims_used == [1, 2]
    assert "search with claim 1 > embed query" in trace.timings
    assert "search with claim 2 > embed query" in trace.timings


# ---------------------------------------------------------------- Batch A: cut-off warning
def test_cut_off_query_claim_produces_warning():
    from patsearch.cli import cut_off_warnings
    patent = Patent("X1", "WHEEL", "", "", ("", "", ""), "", [], [
        Claim(1, "A wheel comprising a semicircular plate (", None, False, True),
        Claim(2, "The wheel of claim 1, wherein the plate is steel.", 1, False, False),
    ])
    assert len(cut_off_warnings(patent, [1])) == 1
    assert "claim 1" in cut_off_warnings(patent, [1])[0]
    assert cut_off_warnings(patent, [2]) == []
