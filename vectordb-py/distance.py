import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine *distance* = 1 - cosine similarity. 0 = same direction, 2 = opposite."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

METRICS = {
    "cosine": cosine,
    "euclidean": euclidean,
    "manhattan": manhattan,
}