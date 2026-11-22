"""Build and load the search index: BM25 inverted index + dense embeddings + metadata."""
from __future__ import annotations

import math
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .data import Patent, Piece, count_glued_paragraphs, load_patents, make_pieces
from .trace import Trace

INDEX_DIR = os.environ.get("PATSEARCH_INDEX", "index")
MODEL_NAME = os.environ.get("PATSEARCH_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_STOP = set(
    "a an the of to in on for and or with by at from as is are be that this which said "
    "wherein comprising comprises claim claims one more each first second being has have "
    "having its it into such than then thereof where when least plurality".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    """Inverted-index BM25. Scoring only touches the posting lists of query terms,
    and accepts an optional candidate mask so filters can be applied *before* scoring."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(lens.mean()) if self.n else 0.0
        self.norm = (k1 * (1 - b + b * lens / max(self.avgdl, 1e-9))).astype(np.float32)
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                post[t].append((i, tf))
        self.postings = {
            t: (np.array([i for i, _ in v], dtype=np.int32), np.array([f for _, f in v], dtype=np.float32))
            for t, v in post.items()
        }
        self.idf = {
            t: math.log(1 + (self.n - len(v[0]) + 0.5) / (len(v[0]) + 0.5)) for t, v in self.postings.items()
        }

    def scores(self, query_tokens: list[str], mask: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            ids, tf = self.postings[t]
            if mask is not None:
                keep = mask[ids]
                ids, tf = ids[keep], tf[keep]
            out[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.norm[ids])
        return out


class Index:
    def __init__(self, patents, pieces, bm25, emb, model_name):
        self.patents: list[Patent] = patents
        self.pieces: list[Piece] = pieces
        self.bm25: BM25 = bm25
        self.emb: np.ndarray = emb  # (n_pieces, dim) float32, L2-normalised
        self.model_name = model_name
        self.by_doc = {p.doc: i for i, p in enumerate(patents)}
        self.piece_pat = np.array([self.by_doc[pc.doc] for pc in pieces], dtype=np.int32)
        self.piece_by_id = {pc.pid: i for i, pc in enumerate(pieces)}
        # patent-level lookups used by the filters
        self.title_abs_tokens = [set(tokenize(f"{p.title} {p.abstract}")) for p in patents]
        self.kw_postings: dict[str, set[int]] = defaultdict(set)
        for i, toks in enumerate(self.title_abs_tokens):
            for t in toks:
                self.kw_postings[t].add(i)
        self.title_norm: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(patents):
            self.title_norm[normalize_title(p.title)].append(i)
        self._model = None

    # ------------------------------------------------------------------ model
    def model_loaded(self) -> bool:
        return self._model is not None

    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_query(self, text: str, prefix: bool = True) -> np.ndarray:
        q = (QUERY_PREFIX + text) if prefix else text
        return self.model().encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)


def normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _embed_text(pc: Piece, title: str) -> str:
    # Prefix the title so short or context-free paragraphs still say what the invention is.
    return f"{title}. {pc.text}" if title and pc.kind != "abstract" else pc.text


def build(data_dir: str | None = None, out_dir: str = INDEX_DIR, model_name: str = MODEL_NAME,
          trace: Trace | None = None) -> Index:
    tr = trace or Trace()
    with tr.step("load + clean JSON files"):
        patents = load_patents(data_dir) if data_dir else load_patents()
        n_empty = sum(1 for p in patents if not p.paragraphs)
        n_fix = sum(1 for p in patents if p.claims and p.claims[0].broken)
        tr.info(f"{len(patents)} patents; {n_empty} with empty description (kept: searched on abstract+claims)")
        tr.info(f"{n_fix} claim-1s had lost their opening words -> rebuilt from title")
        tr.info(f"{len({p.family for p in patents})} families after sibling grouping")
        tr.info(f"{count_glued_paragraphs(patents)} paragraphs still have glued number+verb text (e.g. '10is')")
    with tr.step("split into pieces (abstract / claims / paragraphs)"):
        pieces = make_pieces(patents)
        kinds = Counter(pc.kind for pc in pieces)
        tr.info(f"{len(pieces)} pieces: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    with tr.step("build BM25 inverted index"):
        bm25 = BM25([tokenize(pc.text) for pc in pieces])
        tr.info(f"{len(bm25.postings)} distinct terms, avg piece length {bm25.avgdl:.1f} tokens")
    with tr.step(f"embed pieces with {model_name} (CPU, one-off)"):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        title_of = {p.doc: p.title for p in patents}
        texts = [_embed_text(pc, title_of[pc.doc]) for pc in pieces]
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=tr.verbose).astype(np.float32)
        tr.info(f"matrix {emb.shape}, {emb.nbytes / 1e6:.1f} MB, {len(texts) / (time.perf_counter() - t0):.0f} pieces/s")
    with tr.step(f"save to {out_dir}/"):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "emb.npy"), emb)
        with open(os.path.join(out_dir, "meta.pkl"), "wb") as fh:
            pickle.dump({"patents": patents, "pieces": pieces, "bm25": bm25, "model": model_name}, fh)
    idx = Index(patents, pieces, bm25, emb, model_name)
    idx._model = model
    return idx


def load(index_dir: str = INDEX_DIR, trace: Trace | None = None) -> Index:
    tr = trace or Trace()
    meta_path = os.path.join(index_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No index at {index_dir!r}. Run: python -m patsearch build")
    with tr.step(f"load index from {index_dir}/"):
        with open(meta_path, "rb") as fh:
            m = pickle.load(fh)
        emb = np.load(os.path.join(index_dir, "emb.npy"))
        idx = Index(m["patents"], m["pieces"], m["bm25"], emb, m["model"])
        tr.info(f"{len(idx.patents)} patents, {len(idx.pieces)} pieces, vectors {emb.shape}")
    return idx
