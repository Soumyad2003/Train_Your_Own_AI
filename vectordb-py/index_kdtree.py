import heapq
import itertools
import numpy as np
from distance import METRICS


class _Node:
    __slots__ = ("id", "point", "axis", "left", "right")

    def __init__(self, id, point, axis):
        self.id, self.point, self.axis = id, point, axis
        self.left = self.right = None


class KDTreeIndex:
    def __init__(self, metric: str = "cosine"):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric
        # Cosine is handled as euclidean on unit vectors (see _prep and search).
        self._internal = "euclidean" if metric == "cosine" else metric
        self.dist = METRICS[self._internal]

        self.vectors: dict = {}
        self.root = None
        self._dirty = False
        self.last_dist_calls = 0     # distance computations in the last search

    def _prep(self, v):
        v = np.asarray(v, dtype=np.float32)
        if self.metric == "cosine":
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def insert(self, id, vector) -> None:
        self.vectors[id] = self._prep(vector)
        self._dirty = True

    def delete(self, id) -> bool:
        removed = self.vectors.pop(id, None) is not None
        if removed:
            self._dirty = True
        return removed

    def __len__(self) -> int:
        return len(self.vectors)

    # ---- build ----
    def _rebuild(self) -> None:
        self._ids = list(self.vectors.keys())
        if self._ids:
            self._mat = np.stack([self.vectors[i] for i in self._ids])
            self.root = self._build(np.arange(len(self._ids)))
        else:
            self._mat, self.root = None, None
        self._dirty = False

    def _build(self, idxs):
        if len(idxs) == 0:
            return None
        pts = self._mat[idxs]
        axis = int(np.argmax(pts.var(axis=0)))          # split where the data is most spread out
        order = idxs[np.argsort(pts[:, axis], kind="stable")]
        mid = len(order) // 2                            # median -> balanced tree
        node = _Node(self._ids[order[mid]], self._mat[order[mid]], axis)
        node.left = self._build(order[:mid])
        node.right = self._build(order[mid + 1:])
        return node

    # ---- search ----
    def search(self, query, k: int = 5) -> list[tuple]:
        if self._dirty:
            self._rebuild()
        self.last_dist_calls = 0
        if self.root is None:
            return []

        q = self._prep(query)
        heap = []                       # max-heap of the best k so far: (-dist, tiebreak, id)
        counter = itertools.count()

        def visit(node):
            if node is None:
                return
            d = self.dist(q, node.point)
            self.last_dist_calls += 1
            item = (-d, next(counter), node.id)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, item)

            diff = q[node.axis] - node.point[node.axis]
            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            visit(near)
            # Only cross the splitting plane if it is closer than our current k-th best.
            if len(heap) < k or abs(diff) < -heap[0][0]:
                visit(far)

        visit(self.root)

        results = sorted(((-nd, id) for nd, _, id in heap), key=lambda t: t[0])
        if self.metric == "cosine":
            results = [(d * d / 2.0, id) for d, id in results]   # euclid on unit vectors -> cosine distance
        return [(id, d) for d, id in results]