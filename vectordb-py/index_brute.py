import heapq
import numpy as np
from distance import METRICS

class BruteForceIndex:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.dist = METRICS[metric]
        self.vectors: dict[int | str, np.ndarray] = {}

    def insert(self, id, vector) -> None:
        self.vectors[id] = np.asarray(vector, dtype=np.float32)

    def delete(self, id) -> bool:
        return self.vectors.pop(id, None) is not None

    def search(self, query, k: int = 5) -> list[tuple]:
        """Return [(id, distance), ...] sorted from nearest to farthest."""
        q = np.asarray(query, dtype=np.float32)
        scored = ((self.dist(q, v), id) for id, v in self.vectors.items())
        return [(id, d) for d, id in heapq.nsmallest(k, scored)]

    def __len__(self) -> int:
        return len(self.vectors)