import numpy as np


class Projection:
    """2D PCA of a set of embeddings. Vectors are length-normalised first so the picture
    matches cosine distance, which is what the search uses."""

    def __init__(self, ids, matrix):
        X = np.asarray(matrix, dtype=np.float64)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        n, d = X.shape
        self.ids = list(ids)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean

        if n >= 2:
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:2].copy()
            var = S ** 2
            self.explained = (var[:2] / var.sum()).tolist() if var.sum() > 0 else [0.0, 0.0]
        else:                                  # a single point can't be spread out
            comps = np.zeros((2, d))
            self.explained = [0.0, 0.0]

        # SVD signs are arbitrary; fix them so the plot doesn't flip when documents change
        for i in range(2):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i][j] < 0:
                comps[i] *= -1

        self.components = comps                # shape (2, d)
        self.coords = Xc @ comps.T             # shape (n, 2)

    def project(self, vector):
        v = np.asarray(vector, dtype=np.float64)
        v = v / max(np.linalg.norm(v), 1e-12)
        return ((v - self.mean) @ self.components.T).tolist()