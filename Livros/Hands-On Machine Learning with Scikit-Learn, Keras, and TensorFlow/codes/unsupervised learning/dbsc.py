from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# ==================== DBSCAN ====================
"""
Parameter eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
Parameter min_samples: number of samples in a neighborhood for a point to be considered as a core point
dbscan.labels_: cluster labels
dbscan.core_sample_indices_: core samples
dbscan.components_: core samples and border samples
dbscan.core_sample_indices_: indices of core samples
dbscan.components_: core samples and border samples
dbscan.eps_: eps
dbscan.min_samples_: min_samples
"""

# ==================== Example ====================
eps : float = 0.2

min_samples : int = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

y_pred = dbscan.fit_predict(X)

# ==================== Result ====================
print(y_pred)
print(y_pred is dbscan.labels_)
print(dbscan.core_sample_indices_)
print(dbscan.components_)
print(dbscan.eps)
print(dbscan.min_samples)
# ==================== Visualization ====================
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.show()

# ==================== Decision Boundary ====================
import numpy as np

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = dbscan.fit_predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.show()
