import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MyRidgeRegression:
    def __init__(
            self,
            alpha: float = 1.0
    ):
        """
        Initialize the Ridge Regression model.

        Args:
            alpha (float): The regularization strength.
        """
        self.alpha = alpha
        self.coefficients = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Ridge Regression model to the provided data.

        Args:
            X (np.ndarray): The input features (2D array).
            y (np.ndarray): The target values (1D array).

        The fitting process solves the normal equation:
            coefficients = (X^T X + alpha * I)^-1 X^T y

        Explanation of numpy operations:
        - `np.eye`: Creates a diagonal matrix with ones on the diagonal.
        - `np.linalg.inv`: Computes the inverse of a matrix.
        - `@`: Performs matrix multiplication.
        - `X.T`: Computes the transpose of the matrix X.
        """
        # Generate the identity matrix
        I = np.eye(X.shape[1])
        # Solve the normal equation to find the coefficients
        self.coefficients = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features.

        Args:
            X (np.ndarray): The input features (2D array).

        Returns:
            np.ndarray: The predicted target values.
        """
        return X @ self.coefficients
    
    def get_intercept_and_coef(self):
        return self.coefficients[0], self.coefficients[1:]
    


if __name__ == "__main__":
    # Generate some data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Fit a polynomial regression model
    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    model = MyRidgeRegression(alpha=2.0)
    model.fit(X_poly, y)

    # Generate predictions
    X_range = np.linspace(0, 2, 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    # Plot the data and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data")
    plt.plot(X_range, y_range_pred, color='red', label="Predictions")
    plt.title("Polynomial Ridge Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()