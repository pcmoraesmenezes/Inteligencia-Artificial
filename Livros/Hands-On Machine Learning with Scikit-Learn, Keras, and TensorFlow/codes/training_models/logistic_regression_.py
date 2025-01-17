import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class MyLogisticRegression:
    def __init__(
            self,
            learning_rate: float = 0.1,
            num_iterations: int = 1000
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None

    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))

    def _compute_cost(
            self,
            X: np.array,
            y: np.array
    ) -> float:
        m = X.shape[0]
        h = self.sigmoid(X @ self.theta)
        epsilon = 1e-5
        cost = -1 / m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    def fit(
            self,
            X: np.array,
            y: np.array
    ):
        m, n = X.shape
        self.theta = np.zeros((n, 1))
        y = y.reshape(-1, 1)  

        for _ in range(self.num_iterations):
            h = self.sigmoid(X @ self.theta)
            gradient = X.T @ (h - y) / m
            self.theta -= self.learning_rate * gradient

    def predict_proba(
            self,
            X: np.array
    ) -> np.array:
        return self.sigmoid(X @ self.theta)

    def predict(
            self,
            X: np.array
    ) -> np.array:
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris["data"][:, 3:]
    y = (iris["target"] == 2).astype(int)

    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    model = MyLogisticRegression()
    model.fit(X_bias, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    X_new_bias = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    y_proba = model.predict_proba(X_new_bias)

    y_proba_positive = y_proba.flatten()
    y_proba_negative = 1 - y_proba_positive

    decision_boundary_idx = np.argmin(np.abs(y_proba_positive - y_proba_negative))
    decision_boundary_x = X_new[decision_boundary_idx, 0]
    decision_boundary_y = y_proba_positive[decision_boundary_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(X_new, y_proba_positive, "g-", label="Iris-Virginica (Positive Class)")
    plt.plot(X_new, y_proba_negative, "b--", label="Not Iris-Virginica (Negative Class)")
    plt.axvline(decision_boundary_x, color="red", linestyle=":", label="Decision Boundary")

    plt.annotate(
        "Decision Frontier",
        xy=(decision_boundary_x, decision_boundary_y),
        xytext=(decision_boundary_x + 0.5, decision_boundary_y - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
        color="red"
    )

    plt.xlabel("Petal Width (cm)")
    plt.ylabel("Probability")
    plt.title("Logistic Regression - Probability of Iris-Virginica")
    plt.legend()
    plt.grid(True)
    plt.show()
