from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data
y = iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

export_graphviz(
    clf,
    out_file='tree.dot',
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    filled=True
)