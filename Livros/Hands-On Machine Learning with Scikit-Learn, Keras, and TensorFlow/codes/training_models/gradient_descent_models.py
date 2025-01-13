import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

class GradientDescent:
    def __init__(
            self,
            learning_rate: float = 0.001,
            epochs: int = 100000,
            use_bias: bool = True
    ):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.use_bias: bool = use_bias  # Initialize use_bias here
        self.weights: np.ndarray = None
        self.bias: float = 0
        self.losses: List[float] = []
        self.X_mean: np.ndarray = None
        self.X_std: np.ndarray = None
        self.y_mean: float = None
        self.y_std: float = None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def _normalize_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple:
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean()
        self.y_std = y.std()
        X_normalized = (X - self.X_mean) / self.X_std
        y_normalized = (y - self.y_mean) / self.y_std
        return X_normalized, y_normalized

    def _denormalize_weights_and_bias(self):
        self.weights = self.weights / self.X_std
        if self.use_bias:
            self.bias = self.bias * self.y_std + self.y_mean - np.dot(self.X_mean / self.X_std, self.weights)
        else:
            self.bias = 0


    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        predictions = self._predict(X)
        error = (y - predictions).reshape(-1) 
        gradient_weights = -2 * np.dot(X.T, error) / len(X)
        gradient_bias = -2 * np.mean(error)
        return gradient_weights, gradient_bias

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._normalize_X_y(X, y)
        y = y.flatten()  # Ensure y is 1D
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            gradient_weights, gradient_bias = self._compute_gradient(X, y)
            self.weights -= self.learning_rate * gradient_weights
            if self.use_bias:
                self.bias -= self.learning_rate * gradient_bias
            predictions = self._predict(X)
            loss = self._compute_loss(y, predictions)
            self.losses.append(loss)
        self._denormalize_weights_and_bias()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_normalized = (X - self.X_mean) / self.X_std
        return self._predict(X_normalized)

    def plot_cost_over_time(self) -> None:
        epochs = np.arange(1, len(self.losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.losses, label='Cost Function (Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Cost (Loss)')
        plt.title('Cost Function Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_predict_x_y(self, X: np.ndarray, y: np.ndarray) -> None:
        y_pred = self.predict(X)
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label='True Values')
        plt.plot(X, y_pred, color='red', label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Predictions vs True Values')
        plt.grid(True)
        plt.legend()
        plt.show()

    def _coefficients(self) -> Union[None, tuple]:
        if self.weights is None:
            return None
        return self.weights, self.bias
    
    def __str__(self):
        weights, bias = self._coefficients() if self.weights is not None else ("Not trained", "Not trained")
        return (
            f"Gradient Descent Model:\n"
            f"  Coefficients: Weights={weights}, Bias={bias}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Epochs: {self.epochs}"
        )

    def __repr__(self):
        return f"GradientDescent(learning_rate={self.learning_rate}, epochs={self.epochs}, use_bias={self.use_bias})"


if __name__ == '__main__':
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = GradientDescent(learning_rate=0.1, epochs=10000, use_bias=False)
    model.fit(X, y)

    model.plot_cost_over_time()
    model.plot_predict_x_y(X, y)

    print(model)
