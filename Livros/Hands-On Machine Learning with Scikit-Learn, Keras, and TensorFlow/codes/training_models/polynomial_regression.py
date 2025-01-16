import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MyPolynomialRegression:
    """
    A class for polynomial regression that supports fitting and predicting data
    using polynomial transformations of input features.
    """
    
    def __init__(self, degree):
        """
        Initialize the polynomial regression model.
        
        Args:
            degree (int): The degree of the polynomial to fit.
        """
        self.degree = degree
        self.coefficients = None
    
    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.
        
        Args:
            X (np.ndarray): The input data (1D or 2D array of features).
        
        Returns:
            np.ndarray: A matrix of features including polynomial terms.
        """
        # Start with a column of ones for the bias term (x^0)
        X_poly = np.ones((X.shape[0], 1))
        # Add polynomial terms (x^1, x^2, ..., x^degree)
        for power in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**power))
        return X_poly
    
    def fit(self, X, y):
        """
        Fit the polynomial regression model to the provided data.
        
        Args:
            X (np.ndarray): The input features (1D or 2D array).
            y (np.ndarray): The target values (1D array).
        
        The fitting process solves the normal equation:
            coefficients = (X^T X)^-1 X^T y
        
        Explanation of numpy operations:
        - `np.linalg.inv`: Computes the inverse of a matrix.
        - `@`: Performs matrix multiplication.
        - `X.T`: Computes the transpose of the matrix X.
        """
        # Ensure X is 2D
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        # Solve the normal equation to find the coefficients
        self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    
    def predict(self, X):
        """
        Predict the target values for the given input features.
        
        Args:
            X (np.ndarray): The input features (1D or 2D array).
        
        Returns:
            np.ndarray: The predicted target values.
        """
        # Ensure X is 2D
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        # Compute predictions
        return X_poly @ self.coefficients
    
    def get_coefficients(self):
        """
        Get the coefficients of the polynomial regression model.
        
        Returns:
            np.ndarray: The model's coefficients.
        """
        if self.coefficients is None:
            raise ValueError("Model is not yet fitted. Please fit the model first.")
        return self.coefficients
    
    def plot_data_and_predictions(self, X, y, X_test, y_pred):
        """
        Plot the original data and model predictions.
        
        Args:
            X (np.ndarray): The original input data.
            y (np.ndarray): The original target values.
            X_test (np.ndarray): The test input data for predictions.
            y_pred (np.ndarray): The predicted target values for X_test.
        """
        plt.scatter(X, y, label="Original Data", alpha=0.7)
        plt.plot(X_test, y_pred, label="Model Prediction", color="red")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Polynomial Regression")
        plt.legend()
        plt.show()


def run_my_polynomial_regression_example():
        np.random.seed(0)
        X = np.linspace(0, 10, 100)
        y = 0.5 * X**3 - 2 * X**2 + 3 * X - 1 + np.random.normal(0, 10, 100)

        model = MyPolynomialRegression(degree=3)
        model.fit(X, y)

        X_test = np.linspace(0, 10, 100)
        y_pred = model.predict(X_test)

        model.plot_data_and_predictions(X, y, X_test, y_pred)

        print("Model Coefficients:", model.get_coefficients())

def run_sklearn_polynomial_regression_example():
        np.random.seed(0)
        X = np.linspace(0, 10, 100)
        y = 0.5 * X**3 - 2 * X**2 + 3 * X - 1 + np.random.normal(0, 10, 100)

        poly_features = PolynomialFeatures(degree=3)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))

        model = LinearRegression()
        model.fit(X_poly, y)

        X_test = np.linspace(0, 10, 100)
        X_test_poly = poly_features.transform(X_test.reshape(-1, 1))
        y_pred = model.predict(X_test_poly)

        plt.scatter(X, y, label="Original Data", alpha=0.7)
        plt.plot(X_test, y_pred, label="Model Prediction", color="red")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Polynomial Regression (Sklearn)")
        plt.legend()
        plt.show()

        print("Model Coefficients:", model.coef_)

def plot_comparation():
        np.random.seed(0)
        X = np.linspace(0, 10, 100)
        y = 0.5 * X**3 - 2 * X**2 + 3 * X - 1 + np.random.normal(0, 10, 100)

        model = MyPolynomialRegression(degree=3)
        model.fit(X, y)

        X_test = np.linspace(0, 10, 100)
        y_pred = model.predict(X_test)

        poly_features = PolynomialFeatures(degree=3)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))

        model_sklearn = LinearRegression()
        model_sklearn.fit(X_poly, y)

        X_test = np.linspace(0, 10, 100)
        X_test_poly = poly_features.transform(X_test.reshape(-1, 1))
        y_pred_sklearn = model_sklearn.predict(X_test_poly)

        plt.scatter(X, y, label="Original Data", alpha=0.7)
        plt.plot(X_test, y_pred, label="My Model Prediction", color="red")
        plt.plot(X_test, y_pred_sklearn, label="Sklearn Model Prediction", color="green")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Polynomial Regression")
        plt.legend()
        plt.show()

        print("My Model Coefficients:", model.get_coefficients())
        print("Sklearn Model Coefficients:", model_sklearn.coef_)


if __name__ == "__main__":
    run_my_polynomial_regression_example()
    run_sklearn_polynomial_regression_example()
    plot_comparation()

