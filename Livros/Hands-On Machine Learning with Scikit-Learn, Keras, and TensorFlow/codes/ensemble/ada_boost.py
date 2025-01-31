from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

clf = AdaBoostClassifier(
    DecisionTreeClassifier(
        max_depth=1
    ),
    n_estimators=200,
    learning_rate=0.5,
    algorithm="SAMME.R"
)

scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())