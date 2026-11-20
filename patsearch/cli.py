"""Command line interface.

  python -m patsearch build                       # clean data, build BM25 + embeddings (one-off)
  python -m patsearch search "tire tread groove" --cpc B60C --kw sipe --before 2024-09-01
  python -m patsearch patent 20240051333          # prior art for a patent (its independent claims)
  python -m patsearch patent 20240051333 --claim 3
  python -m patsearch show 20240051333            # metadata + cleaned claims
  python -m patsearch bench                       # timing table: no / narrow / broad filter, pre vs post

Every command prints a numbered trace of its steps to stderr. Add --quiet to hide it.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import textwrap

from rich.console import Console
from rich.table import Table

from . import index as index_mod
from .search import Filters, search, search_by_patent
from .trace import Trace

console = Console()


def _filters(a) -> Filters:
    return Filters(cpc=a.cpc or [], keywords=a.kw or [], title=a.title, before=a.before)


def _print_hits(res, show_evidence: bool = True) -> None:
    if not res.hits:
        console.print("[yellow]No results.[/yellow]")
        return
    t = Table(title=f"Top {len(res.hits)} (strategy={res.strategy}, {res.n_candidates} patents passed filters)",
              show_lines=True)
    for col in ("#", "patent", "title / best matching piece", "cpc", "published", "ranks bm25/emb"):
        t.add_column(col, overflow="fold")
    for i, h in enumerate(res.hits, 1):
        ev = h.evidence[0]
        body = f"[bold]{h.title}[/bold]"
        if show_evidence:
            body += f"\n[cyan]{ev.ref}[/cyan]: {textwrap.shorten(ev.text, 260)}"
        if h.siblings:
            body += f"\n[dim]+ siblings: {', '.join(h.siblings)}[/dim]"
        ranks = f"{ev.bm25_rank or '-'} / {ev.dense_rank or '-'}"
        t.add_row(str(i), h.doc, body, h.cpc, h.pub_date, ranks)
    console.print(t)


def _require_model_lib():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        sys.exit("sentence-transformers is not installed in this Python.\n"
                 "Activate the project environment first:  .venv\\Scripts\\activate  (then re-run)\n"
                 "or install:  pip install -r requirements.txt")


def cmd_build(a):
    _require_model_lib()
    tr = Trace(verbose=not a.quiet)
    index_mod.build(data_dir=a.data, out_dir=a.index, trace=tr)
    console.print(f"[green]Index built in {a.index}/[/green]")


def cmd_search(a):
    if a.mode != "bm25":
        _require_model_lib()
    tr = Trace(verbose=not a.quiet)
    idx = index_mod.load(a.index, trace=tr)
    res = search(idx, a.query, _filters(a), k=a.k, mode=a.mode, force_strategy=a.strategy, trace=tr)
    _print_hits(res)


def cmd_patent(a):
    if a.mode != "bm25":
        _require_model_lib()
    tr = Trace(verbose=not a.quiet)
    idx = index_mod.load(a.index, trace=tr)
    res, claims = search_by_patent(idx, a.doc, claim=a.claim, filters=_filters(a), k=a.k,
                                   date_cutoff=not a.no_date_cutoff, mode=a.mode, trace=tr)
    _print_hits(res)


def cmd_show(a):
    idx = index_mod.load(a.index, trace=Trace(verbose=not a.quiet))
    if a.doc not in idx.by_doc:
        sys.exit(f"Patent {a.doc} not found")
    p = idx.patents[idx.by_doc[a.doc]]
    console.print(f"[bold]{p.doc}[/bold] {p.title}\nCPC {p.cpc_display} (raw {p.cpc}) | published {p.pub_date} | "
                  f"family {p.family} | {len(p.paragraphs)} paragraphs | {len(p.claims)} claims")
    console.print(f"\n[bold]Abstract[/bold]: {p.abstract}\n")
    for c in p.claims:
        flags = " ".join(x for x, on in (("[broken]", c.broken), ("[cut-off]", c.cut_off)) if on)
        dep = f"(depends on {c.depends_on})" if c.depends_on else "(independent)"
        console.print(f"[cyan]claim {c.num}[/cyan] {dep} {flags}: {textwrap.shorten(c.text, 220)}")


def cmd_bench(a):
    """Time search with no / narrow / broad filters, forcing pre- and post-filter strategies."""
    _require_model_lib()
    idx = index_mod.load(a.index, trace=Trace(verbose=not a.quiet))
    import random

    rnd = random.Random(0)
    qs = [p for p in idx.patents if p.claims]
    qs = rnd.sample(qs, min(a.n, len(qs)))
    queries = [p.claims[0].text for p in qs]
    # narrow = a deep code with ~5% of patents, broad = B60B (~half the data)
    setups = [
        ("no filter", Filters(), [None]),
        ("narrow: cpc B60C11/03", Filters(cpc=["B60C11/03"]), ["pre", "post"]),
        ("narrow: keyword 'spoke'", Filters(keywords=["spoke"]), ["pre", "post"]),
        ("broad: cpc B60B", Filters(cpc=["B60B"]), ["pre", "post"]),
    ]
    idx.embed_query("warm up")
    t = Table(title=f"Timing over {len(queries)} claim queries (median ms; query embedding shown separately)")
    for col in ("filter", "strategy", "patents pass", "embed query", "search (excl. embed)", "total",
                "top-10 overlap vs pre"):
        t.add_column(col)
    for name, f, strategies in setups:
        tops: dict[str, list[list[str]]] = {}
        for strat in strategies:
            emb_ms, srch_ms, tot_ms, top = [], [], [], []
            for q in queries:
                tr = Trace()
                r = search(idx, q, f, k=10, force_strategy=strat, trace=tr)
                e = tr.timings.get("embed query", 0.0)
                total = sum(v for kk, v in tr.timings.items())
                emb_ms.append(e)
                srch_ms.append(total - e)
                tot_ms.append(total)
                top.append([h.doc for h in r.hits])
            tops[strat or "none"] = top
            overlap = "-"
            if strat == "post" and "pre" in tops:
                ov = [len(set(x) & set(y)) / max(len(x), 1) for x, y in zip(tops["pre"], top)]
                overlap = f"{statistics.mean(ov):.0%}"
            t.add_row(name, strat or "none", str(r.n_candidates), f"{statistics.median(emb_ms):.1f}",
                      f"{statistics.median(srch_ms):.1f}", f"{statistics.median(tot_ms):.1f}", overlap)
    console.print(t)


def cmd_eval(a):
    """Results table: keyword only / embeddings only / both merged, on the automatic test sets."""
    from . import evaluate as ev

    _require_model_lib()
    tr = Trace(verbose=not a.quiet)
    idx = index_mod.load(a.index, trace=tr)
    idx.embed_query("warm up")
    with tr.step("build test sets"):
        own_tune, own_report = ev.split(ev.own_description_cases(idx, n=a.n))
        sib_tune, sib_report = ev.split(ev.sibling_cases(idx))
        sets = {"own-description": own_tune if a.pile == "tune" else own_report,
                "siblings": sib_tune if a.pile == "tune" else sib_report}
        for name, cs in sets.items():
            tr.info(f"{name}: {len(cs)} queries ({a.pile} pile)")
    t = Table(title=f"Results ({a.pile} pile; own-description is optimistic, see evaluate.py)")
    for col in ("test set", "setup", "queries", "hit@1", "hit@10", "MRR", "avg first-correct rank", "ms/query"):
        t.add_column(col)
    misses = {}
    for name, cases in sets.items():
        with tr.step(f"run {name}"):
            for mode in ev.MODES:
                r = ev.run(idx, cases, mode, trace=tr)
                misses[(name, mode)] = r["misses"]
                label = {"bm25": "keyword only", "dense": "embeddings only", "hybrid": "both merged (RRF)"}[mode]
                t.add_row(name, label, str(r["n"]), f"{r['hit@1']:.0%}", f"{r['hit@10']:.0%}", f"{r['mrr']:.3f}",
                          f"{r['avg_rank']:.1f}", f"{r['ms']:.0f}")
    console.print(t)
    if a.show_misses:
        for (name, mode), m in misses.items():
            if m:
                console.print(f"[yellow]{name} / {mode} misses:[/yellow] {', '.join(x.split(':')[1] for x in m)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="patsearch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=index_mod.INDEX_DIR, help="index directory (default: index)")
    ap.add_argument("--quiet", action="store_true", help="hide the step-by-step trace")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="clean data and build the index")
    b.add_argument("--data", default=None, help="folder with patents_ipa*.json (default: data/raw)")
    b.set_defaults(fn=cmd_build)

    def add_filters(p):
        p.add_argument("--cpc", action="append", help="classification filter, repeatable: B60B, B60B27, B60C11/03")
        p.add_argument("--kw", action="append", help="keyword that must appear in title or abstract (repeatable)")
        p.add_argument("--title", help="exact title")
        p.add_argument("--before", help="only patents published before YYYY-MM-DD")
        p.add_argument("-k", type=int, default=10, help="number of patents to return")
        p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")

    s = sub.add_parser("search", help="search with a text query or a claim")
    s.add_argument("query")
    add_filters(s)
    s.add_argument("--strategy", choices=["pre", "post"], help="force filter strategy (default: planner decides)")
    s.set_defaults(fn=cmd_search)

    p = sub.add_parser("patent", help="prior-art search for a patent in the dataset")
    p.add_argument("doc")
    p.add_argument("--claim", type=int, help="claim number (default: all independent claims)")
    p.add_argument("--no-date-cutoff", action="store_true", help="also return later publications")
    add_filters(p)
    p.set_defaults(fn=cmd_patent)

    sh = sub.add_parser("show", help="show a patent's cleaned metadata and claims")
    sh.add_argument("doc")
    sh.set_defaults(fn=cmd_show)

    be = sub.add_parser("bench", help="timing table for hybrid search")
    be.add_argument("-n", type=int, default=30, help="number of claim queries")
    be.set_defaults(fn=cmd_bench)

    ev = sub.add_parser("eval", help="results table on the automatic test sets")
    ev.add_argument("-n", type=int, default=50, help="own-description queries before the tune/report split")
    ev.add_argument("--pile", choices=["tune", "report"], default="tune",
                    help="tune while developing; report only for the final number")
    ev.add_argument("--show-misses", action="store_true", help="list failed queries (for the weakest-link review)")
    ev.set_defaults(fn=cmd_eval)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
