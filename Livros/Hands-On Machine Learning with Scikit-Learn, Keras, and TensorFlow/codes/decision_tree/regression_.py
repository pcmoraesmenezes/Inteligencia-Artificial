from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, 2:]
y = iris.target

clf = DecisionTreeRegressor()
clf.fit(X, y)

export_graphviz(
    clf,
    out_file='tree.dot',
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
