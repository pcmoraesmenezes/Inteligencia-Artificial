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
    

class MyLassoRegression:
    def __init__(
            self,
            alpha: float = 1.0
    ):
        """
        Initialize the Lasso Regression model.

        Args:
            alpha (float): The regularization strength.
        """
        self.alpha = alpha
        self.coefficients = None

    def _calculate_mse(self, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray) -> float:
        """
        Calculate the mean squared error for the given data and coefficients.

        Args:
            X (np.ndarray): The input features (2D array).
            y (np.ndarray): The target values (1D array).
            coefficients (np.ndarray): The model coefficients.

        Returns:
            float: The mean squared error.
        """
        return np.mean((X @ coefficients - y) ** 2)

    def _compute_cost_function(
            self,
            X: np.ndarray,
            y: np.ndarray,
            coefficients: np.ndarray
    ) -> float:
    
        J = self._calculate_mse(X, y, coefficients) + self.alpha * np.sum(np.abs(coefficients))
        return J
    
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs : int = 1000):
        """
        Fit the Lasso Regression model to the provided data using coordinate descent.

        Args:
            X (np.ndarray): The input features (2D array).
            y (np.ndarray): The target values (1D array).
        """
        # Number of features
        n_features = X.shape[1]
        # Initialize coefficients
        self.coefficients = np.zeros(n_features)

        epochs = 0
        
        while True:
            coefficients_old = self.coefficients.copy()
            for j in range(n_features):
                residual = y - (X @ self.coefficients - self.coefficients[j] * X[:, j])
                rho_j = X[:, j].T @ residual
                if rho_j > self.alpha:
                    self.coefficients[j] = (rho_j - self.alpha) / (X[:, j].T @ X[:, j])
                elif rho_j < -self.alpha:
                    self.coefficients[j] = (rho_j + self.alpha) / (X[:, j].T @ X[:, j])
                else:
                    self.coefficients[j] = 0
            
            if np.linalg.norm(self.coefficients - coefficients_old, ord=2) < 1e-6:
                break

            epochs += 1

            if epochs == n_epochs:
                break

        


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features.

        Args:
            X (np.ndarray): The input features (2D array).

        Returns:
            np.ndarray: The predicted target values.
        """
        return X @ self.coefficients
    

class MyElasticNetRegression:
    def __init__(
            self,
            alpha: float = 1.0,
            r : float = 0.5
    ):
        """
        Initialize the Elastic Net Regression model.

        Args:
            alpha (float): The regularization strength.
            r (float): The mixing parameter.
        """
        self.alpha = alpha
        self.r = r
        self.coefficients = None

    def _calculate_mse(self, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray) -> float:
        """
        Calculate the mean squared error for the given data and coefficients.

        Args:
            X (np.ndarray): The input features (2D array).
            y (np.ndarray): The target values (1D array).
            coefficients (np.ndarray): The model coefficients.

        Returns:
            float: The mean squared error.
        """
        return np.mean((X @ coefficients - y) ** 2)
    
    def _compute_cost_function(
            self,
            X: np.ndarray,
            y: np.ndarray,
            coefficients: np.ndarray
    ) -> float:
    
        J = self._calculate_mse(X, y, coefficients) + self.r * self.alpha * np.sum(np.abs(coefficients)) + 0.5 * (1 - self.r) * self.alpha * np.sum(coefficients ** 2)

        return J


    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 1000):
        """
        Fit the Elastic Net Regression model to the provided data using coordinate descent.

        Args:
            X (np.ndarray): The input features (2D array).
            y (np.ndarray): The target values (1D array).
            n_epochs (int): Maximum number of iterations.
        """
        # Number of features
        n_features = X.shape[1]
        # Initialize coefficients
        self.coefficients = np.zeros(n_features)

        epochs = 0

        while True:
            coefficients_old = self.coefficients.copy()
            for j in range(n_features):
                # Compute residual excluding the current feature
                residual = y - (X @ self.coefficients - self.coefficients[j] * X[:, j])
                # Compute rho_j
                rho_j = X[:, j].T @ residual
                # Compute denominator including L2 regularization
                denominator = (X[:, j].T @ X[:, j]) + (1 - self.r) * self.alpha
                # Update coefficient using soft thresholding
                if rho_j > self.alpha * self.r:
                    self.coefficients[j] = (rho_j - self.alpha * self.r) / denominator
                elif rho_j < -self.alpha * self.r:
                    self.coefficients[j] = (rho_j + self.alpha * self.r) / denominator
                else:
                    self.coefficients[j] = 0

            # Check for convergence
            if np.linalg.norm(self.coefficients - coefficients_old, ord=2) < 1e-6:
                break

            epochs += 1
            if epochs == n_epochs:
                break


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features.

        Args:
            X (np.ndarray): The input features (2D array).

        Returns:
            np.ndarray: The predicted target values.
        """
        return X @ self.coefficients
    

if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    model = MyRidgeRegression(alpha=2.0)
    model.fit(X_poly, y)

    X_range = np.linspace(0, 2, 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data")
    plt.plot(X_range, y_range_pred, color='red', label="Predictions")
    plt.title("Polynomial Ridge Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    my_lasso = MyLassoRegression(alpha=0.1)
    my_lasso.fit(X_poly, y.ravel())  
    y_pred = my_lasso.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data")
    plt.plot(X_range, y_pred, color='red', label="Predictions")
    plt.title("Polynomial Lasso Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    my_elastic_net = MyElasticNetRegression(alpha=0.1, r=0.5)
    my_elastic_net.fit(X_poly, y.ravel())
    y_pred = my_elastic_net.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data")
    plt.plot(X_range, y_pred, color='red', label="Predictions")
    plt.title("Polynomial Elastic Net Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()