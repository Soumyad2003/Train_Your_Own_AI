import time
import numpy as np
from index_brute import BruteForceIndex
from index_hnsw import HNSWIndex

rng = np.random.default_rng(7)
K, N, DIM = 5, 2000, 64
centers = rng.normal(size=(20, DIM))

def make(n):
    return centers[rng.integers(0, 20, n)] + 0.3 * rng.normal(size=(n, DIM))

data, queries = make(N), make(100)

brute = BruteForceIndex("cosine")
variants = {
    "simple":    HNSWIndex("cosine", M=16, ef_construction=100, heuristic=False),
    "heuristic": HNSWIndex("cosine", M=16, ef_construction=100, heuristic=True),
}

for i, v in enumerate(data):
    brute.insert(i, v)
for name, idx in variants.items():
    t0 = time.time()
    for i, v in enumerate(data):
        idx.insert(i, v)
    print(f"{name:10s} built in {time.time() - t0:4.1f}s | layers {len(idx.graph)} "
          f"| orphans on layer 0: {idx.orphan_count()}")

def truths_for(queries):
    return [{i for i, _ in brute.search(q, K)} for q in queries]

def evaluate(idx, queries, truths, ef):
    hits = calls = 0
    for q, t in zip(queries, truths):
        hits += len({i for i, _ in idx.search(q, K, ef=ef)} & t)
        calls += idx.last_dist_calls
    return hits / (K * len(queries)), calls / len(queries)

truths = truths_for(queries)
print(f"\nrecall@{K} (avg distance calls per query; brute force = {N})")
print(f"{'ef':>5} | {'simple':>22} | {'heuristic':>22}")
for ef in (10, 50, 100, 200):
    row = []
    for idx in variants.values():
        rec, calls = evaluate(idx, queries, truths, ef)
        row.append(f"{rec:5.3f} ({calls:4.0f} calls)")
    print(f"{ef:5d} | {row[0]:>22} | {row[1]:>22}")

# deletion: remove 400 vectors from everything
removed = [int(i) for i in rng.permutation(N)[:400]]
removed_set = set(removed)
for i in removed:
    brute.delete(i)
    for idx in variants.values():
        idx.delete(i)
truths = truths_for(queries)
print(f"\nafter deleting {len(removed)} vectors (ef=50):")
for name, idx in variants.items():
    rec, _ = evaluate(idx, queries, truths, 50)
    leaked = any(i in removed_set for q in queries for i, _ in idx.search(q, K, ef=50))
    print(f"  {name:10s} recall {rec:5.3f} | orphans {idx.orphan_count():4d} | deleted ids in results: {leaked}")