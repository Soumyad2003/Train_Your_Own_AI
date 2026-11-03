import heapq
import math
import random
import numpy as np
from distance import METRICS


class HNSWIndex:
    def __init__(self, metric: str = "cosine", M: int = 16, ef_construction: int = 100,
                 ef_search: int = 50, heuristic: bool = True, seed: int = 42):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if M < 2:
            raise ValueError("M must be at least 2")
        self.metric = metric
        self._base_dist = METRICS[metric]
        self.M = M
        self.M0 = 2 * M                      # layer 0 is allowed more links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.heuristic = heuristic
        self.mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        self.vectors: dict = {}              # id -> vector
        self.levels: dict = {}               # id -> top layer of that node
        self.graph: list[dict] = []          # graph[layer][id] -> list of neighbour ids
        self.entry = None
        self.max_level = -1
        self.last_dist_calls = 0

    # ---- helpers ----
    def _prep(self, v):
        v = np.asarray(v, dtype=np.float32)
        if self.metric == "cosine":
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def _dist(self, a, b) -> float:
        self.last_dist_calls += 1
        if self.metric == "cosine":          # vectors are unit length
            return 1.0 - float(np.dot(a, b))
        return self._base_dist(a, b)

    def _random_level(self) -> int:
        return int(-math.log(1.0 - self._rng.random()) * self.mL)

    def __len__(self) -> int:
        return len(self.vectors)

    def _select(self, candidates, m: int) -> list:
        """candidates: [(dist_to_base, id)] sorted nearest-first. Returns up to m ids."""
        if not self.heuristic:
            return [i for _, i in candidates[:m]]

        selected, discarded = [], []
        for d, c in candidates:
            if len(selected) >= m:
                break
            vc = self.vectors[c]
            # keep c only if it is closer to the base than to every neighbour already kept
            if all(d < self._dist(vc, self.vectors[s]) for s in selected):
                selected.append(c)
            else:
                discarded.append(c)
        for c in discarded:                  # top up so nodes keep enough links
            if len(selected) >= m:
                break
            selected.append(c)
        return selected

    def orphan_count(self) -> int:
        """Nodes that no other node links to on layer 0 (searches can't reach them)."""
        if not self.graph:
            return 0
        indeg = {i: 0 for i in self.graph[0]}
        for links in self.graph[0].values():
            for nb in links:
                indeg[nb] += 1
        return sum(1 for i, deg in indeg.items() if deg == 0 and i != self.entry)

    # ---- core: best-first search inside one layer ----
    def _search_layer(self, q, entry_ids, ef: int, layer: int):
        visited = set(entry_ids)
        candidates = []                      # min-heap: closest unexplored first
        results = []                         # max-heap (negated): worst of the best `ef` on top
        for e in entry_ids:
            d = self._dist(q, self.vectors[e])
            heapq.heappush(candidates, (d, e))
            heapq.heappush(results, (-d, e))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            if d_c > -results[0][0]:
                break
            for nb in self.graph[layer].get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(q, self.vectors[nb])
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-nd, id) for nd, id in results)

    # ---- insert ----
    def insert(self, id, vector) -> None:
        if id in self.vectors:
            self.delete(id)

        v = self._prep(vector)
        level = self._random_level()
        self.vectors[id] = v
        self.levels[id] = level
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l][id] = []

        if self.entry is None:
            self.entry, self.max_level = id, level
            return

        ep = [self.entry]
        for l in range(self.max_level, level, -1):
            ep = [self._search_layer(v, ep, 1, l)[0][1]]

        for l in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(v, ep, self.ef_construction, l)
            max_links = self.M0 if l == 0 else self.M
            neighbours = self._select(found, self.M)
            self.graph[l][id] = neighbours
            for n in neighbours:             # reverse link; if the list overflows, re-select
                links = self.graph[l][n]
                links.append(id)
                if len(links) > max_links:
                    nv = self.vectors[n]
                    scored = sorted((self._dist(nv, self.vectors[x]), x) for x in links)
                    self.graph[l][n] = self._select(scored, max_links)
            ep = [nid for _, nid in found]

        if level > self.max_level:
            self.entry, self.max_level = id, level

    # ---- search ----
    def search(self, query, k: int = 5, ef: int | None = None) -> list[tuple]:
        self.last_dist_calls = 0
        if self.entry is None:
            return []
        q = self._prep(query)
        ef = max(ef or self.ef_search, k)

        ep = [self.entry]
        for l in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        found = self._search_layer(q, ep, ef, 0)
        return [(id, d) for d, id in found[:k]]

    # ---- delete ----
    def delete(self, id) -> bool:
        if id not in self.vectors:
            return False

        for l in range(self.levels[id] + 1):
            max_links = self.M0 if l == 0 else self.M
            out = self.graph[l].pop(id)
            for n, links in self.graph[l].items():
                if id in links:
                    links.remove(id)
                    cand = set(links)
                    cand.update(o for o in out if o != n)
                    nv = self.vectors[n]
                    scored = sorted((self._dist(nv, self.vectors[x]), x) for x in cand)
                    links[:] = self._select(scored, max_links)

        del self.vectors[id]
        del self.levels[id]
        while self.graph and not self.graph[-1]:
            self.graph.pop()

        if self.entry == id:
            if not self.vectors:
                self.entry, self.max_level = None, -1
            else:
                self.max_level = max(self.levels.values())
                self.entry = next(i for i, lv in self.levels.items() if lv == self.max_level)
        return True