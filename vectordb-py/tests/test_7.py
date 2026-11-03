import time
import numpy as np
from index_brute import BruteForceIndex
from index_hnsw import HNSWIndex
from document_store import DocumentStore

rng = np.random.default_rng(7)
K = 5


def truths_for(brute, queries):
    return [{i for i, _ in brute.search(q, K)} for q in queries]


def evaluate(hnsw, queries, truths, ef=None):
    """Return (recall@K, average distance computations per query)."""
    hits, calls = 0, 0
    for q, t in zip(queries, truths):
        got = {i for i, _ in hnsw.search(q, K, ef=ef)}
        hits += len(got & t)
        calls += hnsw.last_dist_calls
    return hits / (K * len(queries)), calls / len(queries)


# Part 1: quick sanity check
print("Part 1: sanity check (300 vectors, 16 dims)")
data = rng.normal(size=(300, 16))
brute, hnsw = BruteForceIndex("cosine"), HNSWIndex("cosine")
for i, v in enumerate(data):
    brute.insert(i, v)
    hnsw.insert(i, v)
queries = rng.normal(size=(50, 16))
rec, _ = evaluate(hnsw, queries, truths_for(brute, queries), ef=50)
res = hnsw.search(queries[0], K)
print(f"  recall@{K}: {rec:.3f}")
print("  results sorted nearest-first:", all(res[i][1] <= res[i + 1][1] for i in range(len(res) - 1)))
print(f"  layers: {len(hnsw.graph)}, top layer: {hnsw.max_level}")

# Part 2: recall vs speed, on clustered data (closer to real embeddings)
N, DIM = 2000, 64
centers = rng.normal(size=(20, DIM))

def make(n):
    return centers[rng.integers(0, 20, n)] + 0.3 * rng.normal(size=(n, DIM))

data, queries = make(N), make(100)
brute = BruteForceIndex("cosine")
hnsw = HNSWIndex("cosine", M=16, ef_construction=100)

t0 = time.time()
for i, v in enumerate(data):
    brute.insert(i, v)
    hnsw.insert(i, v)
print(f"\nPart 2: built {N} vectors in {time.time() - t0:.1f}s, layers: {len(hnsw.graph)}")

truths = truths_for(brute, queries)
print(f"  recall@{K} vs ef_search   (brute force = {N} distance calls per query)")
for ef in (5, 10, 20, 50, 100, 200):
    rec, calls = evaluate(hnsw, queries, truths, ef=ef)
    print(f"  ef={ef:4d}   recall {rec:5.3f}   avg {calls:6.0f} calls ({calls / N:5.1%} of N)")

# Part 3: deletion
rng_ids = rng.permutation(N)
removed = set(int(i) for i in rng_ids[:400])
for i in removed:
    brute.delete(i)
    hnsw.delete(i)
truths = truths_for(brute, queries)
leaked = any(i in removed for q in queries for i, _ in hnsw.search(q, K, ef=50))
rec, _ = evaluate(hnsw, queries, truths, ef=50)
print(f"\nPart 3: after deleting {len(removed)} of {N} vectors")
print(f"  size: {len(hnsw)}   deleted ids in results: {leaked}   recall@{K} (ef=50): {rec:.3f}")

# Part 4: plug into the real document store (needs Ollama)
print("\nPart 4: DocumentStore with HNSW")
store = DocumentStore(index=HNSWIndex("cosine"))
store.insert_document("Data Structures",
    "A stack is a last-in first-out data structure. You push items onto the top and pop "
    "them off the top. A queue is first-in first-out.")
store.insert_document("Photosynthesis",
    "Photosynthesis is how plants convert sunlight, water and carbon dioxide into glucose "
    "and oxygen. It happens in the chloroplasts.")
for chunk, dist in store.search("How does a stack work?", k=2):
    print(f"  {dist:.3f}  {chunk.title}")