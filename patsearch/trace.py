"""Step-by-step trace so every command shows what it is doing and how long each step takes.

Printed to stderr, so results on stdout stay clean. Silence with --quiet.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager


class Trace:
    def __init__(self, verbose: bool = False, stream=None):
        self.verbose = verbose
        self.stream = stream or sys.stderr
        self.n = 0
        self.timings: dict[str, float] = {}  # step name -> ms (kept even when quiet, for tests/bench)
        self._depth = 0

    def _out(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=self.stream, flush=True)

    @contextmanager
    def step(self, name: str):
        self.n += 1
        pad = "  " * self._depth
        self._out(f"{pad}[{self.n}] {name} ...")
        self._depth += 1
        t0 = time.perf_counter()
        ok = False
        try:
            yield
            ok = True
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self._depth -= 1
            self.timings[name] = ms
            self._out(f"{pad}    {'done' if ok else 'FAILED'} in {ms:.1f} ms")

    def info(self, msg: str) -> None:
        self._out(f"{'  ' * self._depth}  - {msg}")
