import numpy as np
from index_brute import BruteForceIndex
from index_kdtree import KDTreeIndex
from document_store import DocumentStore

rng = np.random.default_rng(42)

# Part 1: the KD-Tree must return the same distances as brute force
print("Part 1: KD-Tree vs brute force")
for metric in ("euclidean", "manhattan", "cosine"):
    data = rng.normal(size=(300, 8))
    brute, kd = BruteForceIndex(metric), KDTreeIndex(metric)
    for i, v in enumerate(data):
        brute.insert(i, v)
        kd.insert(i, v)

    ok = True
    for _ in range(50):
        q = rng.normal(size=8)
        b = [d for _, d in brute.search(q, 5)]
        t = [d for _, d in kd.search(q, 5)]
        if not np.allclose(b, t, atol=1e-4):
            ok = False
            break

    # delete, then compare again
    brute.delete(0); kd.delete(0)
    q = rng.normal(size=8)
    ok = ok and np.allclose([d for _, d in brute.search(q, 5)],
                            [d for _, d in kd.search(q, 5)], atol=1e-4)
    print(f"  {metric:10s} {'OK' if ok else 'MISMATCH'}")

# Part 2: how much does pruning help as dimensions grow?
N = 5000
print(f"\nPart 2: distance computations per query (N={N}; brute force always does {N})")
for dim in (2, 4, 8, 16, 64, 768):
    data = rng.random((N, dim)).astype(np.float32)
    kd = KDTreeIndex("euclidean")
    for i, v in enumerate(data):
        kd.insert(i, v)
    kd.search(data[0], 5)                       # triggers the build
    calls = []
    for _ in range(20):
        kd.search(rng.random(dim), 5)
        calls.append(kd.last_dist_calls)
    avg = np.mean(calls)
    print(f"  dim={dim:4d}   avg {avg:7.0f} calls   ({avg / N:6.1%} of N)")

# Part 3: plug it into the real document store (needs Ollama)
print("\nPart 3: DocumentStore with the KD-Tree")
store = DocumentStore(index=KDTreeIndex("cosine"))
store.insert_document("Data Structures",
    "A stack is a last-in first-out data structure. You push items onto the top and pop "
    "them off the top. A queue is first-in first-out.")
store.insert_document("Photosynthesis",
    "Photosynthesis is how plants convert sunlight, water and carbon dioxide into glucose "
    "and oxygen. It happens in the chloroplasts.")
for chunk, dist in store.search("How does a stack work?", k=2):
    print(f"  {dist:.3f}  {chunk.title}")