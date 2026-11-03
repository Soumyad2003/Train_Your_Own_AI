from index_brute import BruteForceIndex

# Toy 4D vectors: first two dims = "CS-ness", last two = "food-ness"
data = {
    "binary_tree": [0.9, 0.8, 0.1, 0.0],
    "graph":       [0.8, 0.9, 0.0, 0.1],
    "sushi":       [0.0, 0.1, 0.9, 0.8],
    "pizza":       [0.1, 0.0, 0.8, 0.9],
}
query = [0.9, 0.7, 0.05, 0.05]   # clearly a "CS" query

for metric in ("cosine", "euclidean", "manhattan"):
    idx = BruteForceIndex(metric)
    for name, vec in data.items():
        idx.insert(name, vec)
    print(f"\n[{metric}] top 4:")
    for name, d in idx.search(query, k=4):
        print(f"  {name:12s} {d:.4f}")

# Delete test
idx.delete("binary_tree")
print("\nAfter deleting binary_tree:", len(idx), "items")
print("Top 1 now:", idx.search(query, k=1)[0][0])