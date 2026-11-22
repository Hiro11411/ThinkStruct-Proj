"""Load and clean the weekly patent JSON files, and split patents into searchable pieces.

Missing-field rule: every patent is kept. Patents whose description is empty
(every paragraph is an empty string) are searched on their abstract + claims only.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

DATA_DIR = os.environ.get("PATSEARCH_DATA", os.path.join("data", "raw"))
MAX_WORDS = 350  # split longer paragraphs so they fit a 512-token model

# --------------------------------------------------------------------------- models


@dataclass
class Claim:
    num: int
    text: str
    depends_on: int | None  # parent claim number, None = independent
    broken: bool  # number/opening words lost by the upstream parser
    cut_off: bool  # text ends at an opening bracket "("


@dataclass
class Patent:
    doc: str
    title: str
    abstract: str
    cpc: str  # raw, e.g. "B60B104FI"
    cpc_parsed: tuple[str, str, str]  # ("B60B", "1", "04")
    pub_date: str  # YYYY-MM-DD, taken from the filename (publication date)
    paragraphs: list[tuple[int, str]]  # (paragraph number, text), cleaned
    claims: list[Claim]
    family: str = ""  # sibling group id (filled in later)

    @property
    def cpc_display(self) -> str:
        sc, g, sg = self.cpc_parsed
        return f"{sc} {g}/{sg}" if g else sc


@dataclass
class Piece:
    pid: str  # e.g. "20240051333:c1", ":abs", ":p12"
    doc: str
    kind: str  # "abstract" | "claim" | "para"
    ref: str  # human-readable pinpoint, e.g. "claim 1", "para [0012]"
    text: str


# --------------------------------------------------------------------------- CPC


_CPC_RE = re.compile(r"^([A-H]\d{2}[A-Z])(\d+)?([A-Z]{2})?$")


def parse_cpc(raw: str) -> tuple[str, str, str]:
    """'B60B104FI' -> ('B60B', '1', '04').

    The source drops the slash between main group and subgroup. We assume the
    subgroup is the last two digits (true for every code we inspected, e.g.
    B60B 27/00, B60C 9/20). This is a heuristic and is documented as such.
    """
    raw = raw.strip().upper().replace(" ", "").replace("/", "")
    m = _CPC_RE.match(raw)
    if not m:
        return (raw[:4], "", "")
    sub, digits, _suffix = m.groups()
    digits = digits or ""
    if len(digits) <= 2:
        return (sub, digits, "")
    return (sub, digits[:-2], digits[-2:])


def parse_cpc_query(q: str) -> tuple[str, str, str]:
    """User filter -> components. Accepts 'B60B', 'B60B27', 'B60B 27/00', 'B60B27/00'."""
    q = q.strip().upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])\s*(\d+)?\s*(?:/\s*(\d+))?$", q)
    if not m:
        raise ValueError(f"Unrecognised CPC filter: {q!r} (try B60B, B60B27 or B60B27/00)")
    sub, g, sg = m.groups()
    return (sub, g or "", sg or "")


def cpc_matches(code: tuple[str, str, str], query: tuple[str, str, str]) -> bool:
    """Component-wise match, so 'B60B1' does NOT match B60B 11/00 or B60B 15/00."""
    return all(not q or c == q for c, q in zip(code, query))


# --------------------------------------------------------------------------- cleaning

_FIG_GLUE = re.compile(r"\b(FIGS?\.\s?\d+[A-Z]?)(?=[a-z])")
_CLAIM_HEAD = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?\.\s*(.*)$", re.S)
_DEPENDS = re.compile(r"\bclaims?\s+(\d+)", re.I)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
# Reference numeral glued to the next word: "member 10is", "146are", "3shows", "1denotes".
# Only a fixed list of verbs, so real tokens like "10a", "5mm" or "2nd" are left alone.
_GLUE_VERBS = "is|are|has|have|shows|show|illustrates|includes|denotes|represents|indicates|was|were"
_NUMBER_VERB_GLUE = re.compile(r"\b(\d+[A-Z]?)(?=(?:" + _GLUE_VERBS + r")\b)")
GLUED_TEXT = re.compile(r"\b\d+[A-Z]?(?:" + _GLUE_VERBS + r")\b")  # detector, used for reporting


def count_glued_paragraphs(patents: list[Patent]) -> int:
    """How many cleaned paragraphs still contain number+verb glue (should be 0 after cleaning)."""
    glued_count = 0
    for patent_index in range(len(patents)):
        paragraphs = patents[patent_index].paragraphs
        for paragraph_index in range(len(paragraphs)):
            paragraph_text = paragraphs[paragraph_index][1]
            if GLUED_TEXT.search(paragraph_text):
                glued_count += 1
    return glued_count


def _tidy(text: str) -> str:
    text = _FIG_GLUE.sub(r"\1 ", text)
    text = _NUMBER_VERB_GLUE.sub(r"\1 ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)  # "claim 1 , wherein" -> "claim 1, wherein"
    return re.sub(r"\s+", " ", text).strip()


def _is_scrap(text: str) -> bool:
    """Formula fragments like '0.80≤B/A≤1.25,' that the parser split off a paragraph."""
    return len(re.findall(r"[A-Za-z]{3,}", text)) < 3 and len(text) < 60


def clean_paragraphs(raw: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, p in enumerate(raw or []):
        p = _tidy(p or "")
        if not p:
            continue
        if out and _is_scrap(p):  # glue formula scraps onto the paragraph before
            num, prev = out[-1]
            out[-1] = (num, f"{prev} {p}")
            continue
        out.append((i + 1, p))
    return out


def clean_claims(raw: list[str], title: str) -> list[Claim]:
    claims: list[Claim] = []
    expected = 1
    for item in raw or []:
        item = _tidy(item or "")
        if not item:
            continue
        m = _CLAIM_HEAD.match(item)
        if m:
            start, end, body = int(m.group(1)), m.group(2), m.group(3).strip()
            if "(canceled)" in body.lower() or "(cancelled)" in body.lower():
                expected = (int(end) if end else start) + 1
                continue
            num, broken = start, False
        else:
            # Upstream parser dropped "N . A <thing>, comprising:" -> number from position.
            num, body, broken = expected, item, True
            if num == 1 and title:
                body = f"A {title.lower()}, comprising: {body}"
        dep = _DEPENDS.search(body)
        depends_on = int(dep.group(1)) if dep and int(dep.group(1)) < num else None
        claims.append(Claim(num, body, depends_on, broken, body.rstrip().endswith("(")))
        expected = num + 1
    return claims


def _date_from_filename(fn: str) -> str:
    m = re.search(r"-(\d{4})(\d{2})(\d{2})\.XML$", fn, re.I)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


# --------------------------------------------------------------------------- loading


def load_patents(data_dir: str = DATA_DIR) -> list[Patent]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No patents_ipa*.json files in {data_dir!r}. Set PATSEARCH_DATA.")
    patents: list[Patent] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for raw in json.load(fh):
                title = _tidy(raw.get("title") or "")
                cpc = (raw.get("classification") or "").strip()
                patents.append(
                    Patent(
                        doc=str(raw.get("doc_number") or "").strip(),
                        title=title,
                        abstract=_tidy(raw.get("abstract") or ""),
                        cpc=cpc,
                        cpc_parsed=parse_cpc(cpc) if cpc else ("", "", ""),
                        pub_date=_date_from_filename(raw.get("filename") or ""),
                        paragraphs=clean_paragraphs(raw.get("detailed_description") or []),
                        claims=clean_claims(raw.get("claims") or [], title),
                    )
                )
    _assign_families(patents)
    return patents


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _assign_families(patents: list[Patent], threshold: float = 0.6) -> None:
    """Group near-duplicate siblings (same applicant, near-identical abstracts)."""
    sets = [_words(p.abstract) for p in patents]
    parent = list(range(len(patents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(patents)):
        for j in range(i + 1, len(patents)):
            a, b = sets[i], sets[j]
            if a and b and len(a & b) / len(a | b) >= threshold:
                parent[find(j)] = find(i)
    for i, p in enumerate(patents):
        p.family = patents[find(i)].doc


# --------------------------------------------------------------------------- pieces


def _split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def make_pieces(patents: list[Patent]) -> list[Piece]:
    pieces: list[Piece] = []
    for p in patents:
        if p.abstract:
            pieces.append(Piece(f"{p.doc}:abs", p.doc, "abstract", "abstract", p.abstract))
        for c in p.claims:
            pieces.append(Piece(f"{p.doc}:c{c.num}", p.doc, "claim", f"claim {c.num}", c.text))
        for num, text in p.paragraphs:
            for k, chunk in enumerate(_split_long(text)):
                suffix = f".{k + 1}" if k else ""
                pieces.append(
                    Piece(f"{p.doc}:p{num}{suffix}", p.doc, "para", f"para [{num:04d}]{suffix}", chunk)
                )
    return pieces


def claim_with_parents(p: Patent, num: int) -> str:
    """Dependent claim = parent chain + own text (claim N inherits every part of its parent).

    Broken claims lost their 'according to claim X', so their parent is unknown and
    they are treated as standalone.
    """
    by_num = {c.num: c for c in p.claims}
    chain, seen, cur = [], set(), by_num.get(num)
    while cur and cur.num not in seen:
        seen.add(cur.num)
        chain.append(cur.text)
        cur = by_num.get(cur.depends_on) if cur.depends_on else None
    return " ".join(reversed(chain))
