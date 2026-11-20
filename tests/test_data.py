"""Cleaning and parsing rules (no model needed)."""
from patsearch.data import (
    Claim, Patent, claim_with_parents, clean_claims, clean_paragraphs, cpc_matches, make_pieces,
    parse_cpc, parse_cpc_query,
)


# ---------------------------------------------------------------- CPC codes
def test_parse_cpc_drops_missing_slash():
    assert parse_cpc("B60B104FI") == ("B60B", "1", "04")
    assert parse_cpc("B60B2700FI") == ("B60B", "27", "00")
    assert parse_cpc("B60C1103FI") == ("B60C", "11", "03")


def test_cpc_filter_does_not_prefix_match_wrong_group():
    """The naive prefix bug: 'B60B1' would also match B60B 11/.., 15/.., 19/..."""
    q = parse_cpc_query("B60B1")
    assert cpc_matches(parse_cpc("B60B104FI"), q)
    assert not cpc_matches(parse_cpc("B60B1110FI"), q)
    assert not cpc_matches(parse_cpc("B60B1500FI"), q)


def test_cpc_query_formats():
    assert parse_cpc_query("b60b") == ("B60B", "", "")
    assert parse_cpc_query("B60C 11/03") == ("B60C", "11", "03")
    assert parse_cpc_query("B60C11/03") == ("B60C", "11", "03")


# ---------------------------------------------------------------- paragraphs
def test_empty_paragraphs_dropped_and_numbers_kept():
    out = clean_paragraphs(["", "First real paragraph about a wheel hub.", "  ", "Second one about spokes."])
    assert out == [(2, "First real paragraph about a wheel hub."), (4, "Second one about spokes.")]


def test_glued_fig_text_is_split():
    out = clean_paragraphs(["FIG.1is a schematic view of the wheel."])
    assert out[0][1].startswith("FIG.1 is a schematic")


def test_formula_scrap_merged_into_previous_paragraph():
    out = clean_paragraphs(["The ratio of the widths satisfies the following relation:", "0.80≤B/A≤1.25,"])
    assert len(out) == 1 and out[0][1].endswith("0.80≤B/A≤1.25,")


# ---------------------------------------------------------------- claims
def test_claim_numbers_read_from_text_and_canceled_ranges_skipped():
    raw = ["1 - 5 . (canceled)", "6 . A wheel comprising a rim.", "7 . The wheel of claim 6 , wherein the rim is steel."]
    cs = clean_claims(raw, "WHEEL")
    assert [c.num for c in cs] == [6, 7]
    assert cs[1].depends_on == 6 and cs[0].depends_on is None
    assert "claim 6, wherein" in cs[1].text  # stray space before comma removed


def test_broken_claim_one_gets_title_preamble():
    cs = clean_claims(["an axle body having a middle segment;", "2 . The spoke of claim 1 , wherein x."], "SPOKE")
    assert cs[0].broken and cs[0].num == 1
    assert cs[0].text.startswith("A spoke, comprising: an axle body")
    assert cs[1].num == 2 and cs[1].depends_on == 1 and not cs[1].broken


def test_cut_off_claim_flagged():
    cs = clean_claims(["1 . A wheel comprising a semicircular plate ("], "WHEEL")
    assert cs[0].cut_off


def test_dependent_claim_includes_parent_chain():
    p = Patent("X", "T", "", "", ("", "", ""), "", [], [
        Claim(1, "A wheel comprising a rim.", None, False, False),
        Claim(2, "The wheel of claim 1, wherein the rim is steel.", 1, False, False),
        Claim(3, "The wheel of claim 2, wherein the steel is coated.", 2, False, False),
    ])
    txt = claim_with_parents(p, 3)
    assert txt.index("A wheel comprising") < txt.index("rim is steel") < txt.index("steel is coated")


# ---------------------------------------------------------------- pieces
def test_patent_without_description_still_searchable():
    p = Patent("X", "T", "An abstract.", "B60B104FI", ("B60B", "1", "04"), "2024-01-01", [],
               [Claim(1, "A wheel.", None, False, False)])
    kinds = sorted(pc.kind for pc in make_pieces([p]))
    assert kinds == ["abstract", "claim"]


def test_long_paragraph_split():
    long = " ".join(["word"] * 800)
    p = Patent("X", "T", "", "", ("", "", ""), "", [(3, long)], [])
    paras = [pc for pc in make_pieces([p]) if pc.kind == "para"]
    assert len(paras) == 3 and paras[0].ref == "para [0003]" and paras[1].ref == "para [0003].2"
