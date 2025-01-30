from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)

X, y = make_moons(n_samples=1000, noise=0.35)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))