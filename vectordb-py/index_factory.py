from index_brute import BruteForceIndex
from index_kdtree import KDTreeIndex
from index_hnsw import HNSWIndex

INDEX_TYPES = {
    "brute": BruteForceIndex,
    "kdtree": KDTreeIndex,
    "hnsw": HNSWIndex,
}

def make_index(kind: str, metric: str = "cosine"):
    if kind not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {kind}")
    return INDEX_TYPES[kind](metric)