import time

import numpy as np

from index_brute import BruteForceIndex
from index_kdtree import KDTreeIndex
from index_hnsw import HNSWIndex

K = 5
N_QUERIES = 50
_cache: dict = {}          # (n, dim) -> built indexes; avoids rebuilding on every run


def _make_data(n: int, dim: int, n_queries: int, seed: int = 123):
    """Clustered vectors, a bit like real embeddings: 20 topics with noise around each."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, dim))

    def make(m):
        return centers[rng.integers(0, 20, m)] + 0.3 * rng.normal(size=(m, dim))

    return make(n).astype(np.float32), make(n_queries).astype(np.float32)


class _NumpyBrute:
    """Exact cosine search done as one matrix-vector product."""

    def __init__(self, data):
        self.unit = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)

    def search(self, q, k):
        qn = q / max(float(np.linalg.norm(q)), 1e-12)
        d = 1.0 - self.unit @ qn
        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]
        return [(int(i), float(d[i])) for i in idx]


def _measure(search_fn, queries, truth, calls_fn):
    times, hits, calls = [], 0, []
    for q, t in zip(queries, truth):
        t0 = time.perf_counter()
        res = search_fn(q)
        times.append(time.perf_counter() - t0)
        hits += len({i for i, _ in res} & t)
        calls.append(calls_fn())
    return {
        "query_us": float(np.median(times) * 1e6),
        "recall": hits / (K * len(queries)),
        "dist_calls": float(np.mean(calls)),
    }


def _build(n: int, dim: int) -> dict:
    data, queries = _make_data(n, dim, N_QUERIES)

    # 1. Brute force, pure Python loop. Its results are the ground truth for recall.
    t0 = time.perf_counter()
    brute = BruteForceIndex("cosine")
    for i, v in enumerate(data):
        brute.insert(i, v)
    brute_build = (time.perf_counter() - t0) * 1000

    truth, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = brute.search(q, K)
        times.append(time.perf_counter() - t0)
        truth.append({i for i, _ in res})
    brute_row = {
        "name": "Brute force (Python loop)", "build_ms": brute_build,
        "query_us": float(np.median(times) * 1e6), "recall": 1.0, "dist_calls": float(n),
    }

    # 2. Brute force, NumPy
    t0 = time.perf_counter()
    npb = _NumpyBrute(data)
    np_build = (time.perf_counter() - t0) * 1000
    np_row = {
        "name": "Brute force (NumPy)", "build_ms": np_build,
        **_measure(lambda q: npb.search(q, K), queries, truth, lambda: n),
    }

    # 3. KD-Tree
    t0 = time.perf_counter()
    kd = KDTreeIndex("cosine")
    for i, v in enumerate(data):
        kd.insert(i, v)
    kd.search(queries[0], K)               # the tree is built lazily here, so count it as build time
    kd_build = (time.perf_counter() - t0) * 1000
    kd_row = {
        "name": "KD-Tree", "build_ms": kd_build,
        **_measure(lambda q: kd.search(q, K), queries, truth, lambda: kd.last_dist_calls),
    }

    # 4. HNSW (its row is measured per request, because ef can change)
    t0 = time.perf_counter()
    hnsw = HNSWIndex("cosine")
    for i, v in enumerate(data):
        hnsw.insert(i, v)
    hnsw_build = (time.perf_counter() - t0) * 1000

    return {
        "queries": queries, "truth": truth, "hnsw": hnsw, "hnsw_build_ms": hnsw_build,
        "static_rows": [brute_row, np_row, kd_row],
    }


def run_benchmark(n: int, dim: int, ef: int) -> dict:
    key = (n, dim)
    cached = key in _cache
    if not cached:
        if len(_cache) >= 3:                           # keep memory bounded
            _cache.pop(next(iter(_cache)))
        _cache[key] = _build(n, dim)
    b = _cache[key]

    hnsw = b["hnsw"]
    hnsw_row = {
        "name": f"HNSW (ef={ef})", "build_ms": b["hnsw_build_ms"],
        **_measure(lambda q: hnsw.search(q, K, ef=ef), b["queries"], b["truth"],
                   lambda: hnsw.last_dist_calls),
    }
    return {
        "n": n, "dim": dim, "k": K, "ef": ef, "queries": len(b["queries"]),
        "cached": cached, "results": [*b["static_rows"], hnsw_row],
    }