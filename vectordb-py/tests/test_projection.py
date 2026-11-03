import numpy as np
from projection import Projection

rng = np.random.default_rng(1)
X = rng.normal(size=(10, 32))
p = Projection(list(range(10)), X)
print("projecting a stored vector reproduces its coordinates:", np.allclose(p.project(X[3]), p.coords[3]))
print("variance explained by PC1, PC2:", [round(v, 3) for v in p.explained])