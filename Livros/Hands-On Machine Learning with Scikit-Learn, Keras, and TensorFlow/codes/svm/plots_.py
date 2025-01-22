import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  
y = (iris["target"] == 2).astype(float)


def plot_decision_boundary(clf, X, y, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X_new).reshape(x0.shape)
    y_decision = clf.decision_function(X_new).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Not Iris-Virginica")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
    plt.axis(axes)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
])

svm_clf.fit(X, y)

plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plot_decision_boundary(svm_clf, X, y, [0, 7.5, 0, 3])
plt.title(r"$C = {}$".format(svm_clf.named_steps["linear_svc"].C), fontsize=16)
plt.ylabel("")

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=100, loss="hinge", random_state=42)),
])

svm_clf.fit(X, y)

plt.subplot(122)
plot_decision_boundary(svm_clf, X, y, [0, 7.5, 0, 3])
plt.title(r"$C = {}$".format(svm_clf.named_steps["linear_svc"].C), fontsize=16)
plt.show()