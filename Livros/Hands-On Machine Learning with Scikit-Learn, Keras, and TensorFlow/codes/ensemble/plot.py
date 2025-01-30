from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# realizar um plot de uma fronteira de decisão onde eu tenho no gráfico a esquerda a fronteira de decisão da arvore de decisão para o conjunto de dados da lua e na direita a fronteira de decisão do bagging com 500 arvores de decisão

X, y = make_moons(n_samples=1000, noise=0.35)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)



def plot_decision_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_background = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_points = ListedColormap(["#FF0000", "#0000FF"])
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor="k", s=20)
    ax.set_title(title)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)

plot_decision_boundary(tree_clf, X, y, axes[0], "Decision Tree")

plot_decision_boundary(bag_clf, X, y, axes[1], "Bagging (500 Trees)")

plt.show()

