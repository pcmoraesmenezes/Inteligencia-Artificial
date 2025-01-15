import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from time import time
from typing import Tuple

class MyLinearRegression():
    """
    A custom implementation of a simple Linear Regression model.
    
    This class implements Linear Regression using the Normal Equation method.
    It includes methods for training, predicting, evaluating using Mean Squared Error (MSE),
    generating synthetic data, and visualizing the results.

    Attributes:
        coef_ (numpy.ndarray): The coefficients (slopes) of the trained model.
        intercept_ (float): The intercept (bias) of the trained model.
        theta (numpy.ndarray): The parameter vector containing both the intercept and coefficients.
        training_time (float): The time taken to train the model (in seconds).
        prediction_time (float): The time taken to make predictions (in seconds).
        noise (float): The noise level used during data generation (if applicable).
    """
    def __init__(self):
        """
        Initializes the Linear Regression model with default values for attributes.
        """
        self.coef_ = None
        self.intercept_ = None
        self.theta = None
        self.training_time = None
        self.prediction_time = None
        self.noise = None  
        self.costs = []

    def fit(self, X, y):
        """
        Trains the Linear Regression model using the Normal Equation method.
        
        The Normal Equation calculates the optimal parameters (theta) that minimize
        the cost function (Mean Squared Error) for the given training data.

        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, n_features).
            y (numpy.ndarray): The target values of shape (n_samples, 1).
        
        Process:
            1. Augments the input matrix `X` with a column of ones to represent the intercept term.
            2. Computes `theta` using the formula:
                theta = (X_b^T * X_b)^(-1) * X_b^T * y
            where:
                - X_b is the augmented matrix of features.
                - X_b^T is the transpose of X_b.
                - (X_b^T * X_b)^(-1) is the inverse of the product of X_b^T and X_b.
        """
        start_time = time()  # Start timing
        X_b = np.c_[np.ones((len(X), 1)), X]  # Add a column of ones to X
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        # np.linalg.inv: Computes the inverse of a square matrix.
        # .T: Transposes the matrix.
        # .dot: Performs matrix multiplication.
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        end_time = time()  # End timing
        self.training_time = end_time - start_time


    def predict(self, X):
        """
        Makes predictions using the trained Linear Regression model.
        
        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted values of shape (n_samples, 1).

        Process:
            1. Augments the input matrix `X` with a column of ones to account for the intercept term.
            2. Computes predictions as:
                predictions = X_b * theta
            where:
                - X_b is the augmented matrix of features.
                - theta is the parameter vector from training.
        """
        start_time = time()  
        X_b = np.c_[np.ones((len(X), 1)), X]  # Add a column of ones to X
        predictions = X_b.dot(self.theta)  # Compute predictions
        # .dot: Performs matrix multiplication.
        end_time = time()  
        self.prediction_time = end_time - start_time
        return predictions
    
    def mse(self, X, y):
        """
        Computes the Mean Squared Error (MSE) for the predictions.

        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, n_features).
            y (numpy.ndarray): The true target values of shape (n_samples, 1).

        Returns:
            float: The Mean Squared Error.

        Process:
            1. Predicts the target values for the input features.
            2. Computes the squared differences between predicted and true values.
            3. Averages these squared differences.
        """
        return np.mean((self.predict(X) - y) ** 2)
        # np.mean: Computes the mean of an array.

    def gen_data(self, samples, noise):
        """
        Generates synthetic data for testing and demonstration.

        Parameters:
            samples (int): Number of data points to generate.
            noise (float): Standard deviation of the Gaussian noise added to the target values.

        Returns:
            tuple: A tuple (X, y), where:
                - X (numpy.ndarray): Generated input features of shape (samples, 1).
                - y (numpy.ndarray): Generated target values of shape (samples, 1).

        Process:
            1. Generates random values for X in the range [0, 2].
            2. Computes y as a linear function of X with added Gaussian noise:
                y = 4 + 3 * X + noise
        """

        self.noise = noise

        X = 2 * np.random.rand(samples, 1)  # Generate random X values
        # np.random.rand: Generates random samples from a uniform distribution [0, 1).
        y = 4 + 3 * X + np.random.randn(samples, 1) * noise
        # np.random.randn: Generates random samples from a standard normal distribution.
        return X, y
    
    def plot_data(self, X, y):
        """
        Plots the generated data points.

        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, 1).
            y (numpy.ndarray): The target values of shape (n_samples, 1).
        """
        plt.plot(X, y, "b.")  # Blue points for data
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.axis([0, 2, 0, 15])  # Set axis limits
        plt.show()

    def plot_line(self, X, y):
        """
        Plots the data points and the regression line.

        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, 1).
            y (numpy.ndarray): The target values of shape (n_samples, 1).

        Process:
            1. Predicts the target values for two extreme X values (e.g., 0 and 2).
            2. Draws the regression line connecting these predicted values.
        """
        plt.plot(X, y, "b.")  # Blue points for data
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.axis([0, 2, 0, 15])  # Set axis limits
        X_new = np.array([[0], [2]])  # Extreme values for X
        X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add column of ones
        y_predict = X_new_b.dot(self.theta)  # Predict target values
        plt.plot(X_new, y_predict, "r-")  # Red line for regression
        plt.show()

    def get_times(self) -> Tuple[float, float]:
        """
        Returns the training and prediction times of the model.

        Returns:
            tuple: A tuple (training_time, prediction_time), where:
                - training_time (float): Time taken for training (in seconds).
                - prediction_time (float): Time taken for predictions (in seconds).
        """

        training = self.training_time
        prediction = self.prediction_time

        return training, prediction


    def get_intercept_and_coef(self) -> Tuple[float, np.ndarray]:
        """
        Returns the intercept and coefficients of the trained model.

        Returns:
            tuple: A tuple (intercept, coefficients), where:
                - intercept (float): The intercept term of the model.
                - coefficients (numpy.ndarray): The coefficients of the model.
        """
        return self.intercept_, self.coef_
    
    def simulate_cost_over_time(self, X, y, epochs=100):
        """
        Simula o custo ao longo das épocas para a regressão linear usando interpolação linear.
        
        Parameters:
            X (numpy.ndarray): The input features of shape (n_samples, n_features).
            y (numpy.ndarray): The true target values of shape (n_samples, 1).
            epochs (int): The number of epochs to simulate.

        Returns:
            list: A list of simulated costs for each epoch.
        """
        X_b = np.c_[np.ones((len(X), 1)), X]  # Add a column of ones to X
        
        initial_theta = np.random.randn(X_b.shape[1], 1)
        initial_cost = np.mean((X_b.dot(initial_theta) - y) ** 2)

        final_cost = self.mse(X, y)

        simulated_costs = np.linspace(initial_cost, final_cost, epochs).tolist()
        return simulated_costs

    def __str__(self):
        """
        Provides a string representation of the trained model, including the intercept,
        coefficients, equation, and timings.

        Returns:
            str: Description of the model.
        """
        if self.theta is None:
            return "Model not trained yet"

        equation = str(self.intercept_) + " + " + " + ".join([str(c) + "x" + str(i) for i, c in enumerate(self.coef_)])
        equation = equation.replace('[' , '').replace(']' , '')

        training_time_str = f"Training time: {self.training_time:.6f} seconds" if self.training_time else "Training not done"
        prediction_time_str = f"Prediction time: {self.prediction_time:.6f} seconds" if self.prediction_time else "Prediction not done"

        return (
            "=====================\n" +
            "Model: Linear Regression\n" +
            "Intercept: " + str(self.intercept_).replace("[", "").replace("]", "") + "\n" +
            "Coef: " + str(self.coef_).replace("[", "").replace("]", "") + "\n" +
            "Equation: " + equation.replace("+ -", "- ") + "\n" +
            f'Real equation: 4 + 3x * {self.noise}\n' +

            training_time_str + "\n" +
            prediction_time_str + "\n" +
            "====================="
        )

    def __repr__(self):
        """
        Provides a concise representation of the model.
        """
        return self.__str__()

        

# Testing the class
if __name__ == "__main__":
    lr = MyLinearRegression()
    X, y = lr.gen_data(100, 1.3)
    lr.fit(X, y)

    predictions = lr.predict(X)
    print(lr)
    lr.plot_line(X, y)

    # Comparing with sklearn
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print("Sklearn")
    print("Coefficients:", lin_reg.coef_, "Intercept:", lin_reg.intercept_)
