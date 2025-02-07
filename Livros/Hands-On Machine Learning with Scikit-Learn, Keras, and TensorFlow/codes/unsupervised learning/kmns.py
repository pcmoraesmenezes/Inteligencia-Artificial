from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# ==================== KMeans ====================
"""

Parameter k: number of clusters
Parameter init: initialization method. Set a high value for n_init to avoid local minima
kmeans.labels_: cluster labels
kmeans.cluster_centers_: cluster centers
kmeans.inertia_: sum of squared distances from each sample to its closest cluster center
kmeans.score(X): negative inertia

"""

# ==================== Example ====================

k : int = 5
kmeans = KMeans(n_clusters=k)

X, y = make_blobs(n_samples=1000, centers=k, n_features=2, random_state=42)

y_pred = kmeans.fit_predict(X)

# ==================== Result ====================
print(y_pred)
print(y_pred is kmeans.labels_)
print(kmeans.cluster_centers_)

# ==================== Visualization ====================
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

# ==================== Decision Boundary ====================
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

# ==================== Good Initialization ====================
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-3, 0], [-3, -1]])
kmeans = KMeans(n_clusters=k, init=good_init, n_init=1)
# kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)

# ==================== Elbow Method ====================
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.annotate('Elbow', xy=(3, inertia[2]), xytext=(4, inertia[3]),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# ==================== Silhouette Score ====================
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)

# ==================== Silhouette Diagram ====================

from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

silhouette_values = silhouette_samples(X, kmeans.labels_)
y_lower = 10

for i in range(k):
    cluster_silhouette_values = silhouette_values[kmeans.labels_ == i]
    cluster_silhouette_values.sort()
    y_upper = y_lower + cluster_silhouette_values.shape[0]

    color = cm.nipy_spectral(float(i) / k)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    y_lower = y_upper + 10

plt.show()

# ====================  K-means Image Segmentation ====================

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50)),
    ('log_reg', LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

