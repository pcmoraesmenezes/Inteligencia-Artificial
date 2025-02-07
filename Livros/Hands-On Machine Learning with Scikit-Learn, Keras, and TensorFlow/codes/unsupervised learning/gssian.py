from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# ==================== Gaussian Mixture Model ====================
"""
Parameter n_components: number of Gaussian components
gmm.weights_: weights of each Gaussian component
gmm.means_: means of each Gaussian component
gmm.covariances_: covariances of each Gaussian component
gmm.converged_: True if the algorithm converged
gmm.predict_proba(X): posterior probabilities of each component given the data
gmm.predict(X): predicted labels
gmm.score_samples(X): log-likelihood of each sample
"""

# ==================== Example ====================

n_comp : int = 3
n_init : int = 10

gmm = GaussianMixture(n_components=n_comp, n_init=n_init)

X, y = load_iris(return_X_y=True)
X = X[:, :2]

y_pred = gmm.fit_predict(X)

# ==================== Result ====================
print(y_pred)
print(gmm.weights_)
print(gmm.means_)
print(gmm.covariances_)
print(gmm.converged_)
print(gmm.predict_proba(X))
print(gmm.predict(X))
print(gmm.score_samples(X))

# ==================== Visualization ====================
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.show()

# ==================== Decision Boundary ====================
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])  # Now compatible with 2D input
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundaries of GMM Clustering")
plt.show()