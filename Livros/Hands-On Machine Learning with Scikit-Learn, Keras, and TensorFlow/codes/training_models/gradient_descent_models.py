import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from linear_regression import MyLinearRegression
from time import time

class GradientDescent:
    def __init__(
            self,
            learning_rate : float = 0.1,
            epochs : int = 1000
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.costs = []
        self.theta = None
        self.time_training_start = None
        self.time_training_end = None
        self.time_predict_start = None
        self.time_predict_end = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.time_training_start = time()
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(2, 1)

        for epoch in range(self.epochs):
            gradients = 2 / len(X) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta = self.theta - self.learning_rate * gradients
            cost = np.mean((X_b.dot(self.theta) - y) ** 2)
            self.costs.append(cost)

        self.time_training_end = time()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.time_predict_start = time()
        X_b = np.c_[np.ones((len(X), 1)), X]
        predictions = X_b.dot(self.theta)
        self.time_predict_end = time()
        return predictions

    
    def get_times(self):
        training : float = self.time_training_end - self.time_training_start
        prediction : float = self.time_predict_end - self.time_predict_start
        return training, prediction
    
    def get_intercept_and_coef(self):
        return self.theta[0], self.theta[1]
    
    def plot_cost_over_time(self):
        plt.plot(self.costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()

    def plot_predict_x_y(self, X: np.ndarray, y: np.ndarray):
        plt.scatter(X, y)
        plt.plot(X, self.predict(X), color='red')
        plt.show()


class StoachasticGradientDescent:
    def __init__(
            self,
            learning_rate : float = 0.1,
            epochs : int = 1000
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.costs = []
        self.theta = None
        self.time_training_start = None
        self.time_training_end = None
        self.time_predict_start = None
        self.time_predict_end = None
    
    
    def fit(
            self,
            X : np.ndarray,
            y : np.ndarray
    ):
        self.time_training_start = time()

        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(2, 1)
        
        for epoch in range(self.epochs):
            for i in range(len(X)):
                random_index = np.random.randint(len(X))
                xi = X_b[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradients
                cost = np.mean((X_b.dot(self.theta) - y) ** 2)
                self.costs.append(cost)

        self.time_training_end = time()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.time_predict_start = time()
        X_b = np.c_[np.ones((len(X), 1)), X]
        predictions = X_b.dot(self.theta)
        self.time_predict_end = time()
        return predictions

    
    def get_times(self):
        training : float = self.time_training_end - self.time_training_start
        prediction : float = self.time_predict_end - self.time_predict_start
        return training, prediction
    
    def get_intercept_and_coef(self):
        return self.theta[0], self.theta[1]

    def plot_cost_over_time(self):
        plt.plot(self.costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()

    def plot_predict_x_y(self, X: np.ndarray, y: np.ndarray):
        plt.scatter(X, y)
        plt.plot(X, self.predict(X), color='red')
        plt.show()


class MiniBatchGradientDescent:
    def __init__(
            self,
            learning_rate: float = 0.1,
            epochs: int = 1000,
            batch_size: int = 10
    ):
    
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.costs = []
        self.theta = None
        self.batch_size = batch_size
        self.time_training_start = None
        self.time_training_end = None
        self.time_predict_start = None
        self.time_predict_end = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ): 
        self.time_training_start = time()
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(2, 1)

        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(len(X))
            X_b_shuffled = X_b[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, len(X), self.batch_size):
                xi = X_b_shuffled[i:i+self.batch_size]
                yi = y_shuffled[i:i+self.batch_size]
                gradients = 2 / self.batch_size * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradients
                cost = np.mean((X_b.dot(self.theta) - y) ** 2)
                self.costs.append(cost)
            
        self.time_training_end = time()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.time_predict_start = time()
        X_b = np.c_[np.ones((len(X), 1)), X]
        predictions = X_b.dot(self.theta)
        self.time_predict_end = time()
        return predictions

    def get_times(self):
        training : float = self.time_training_end - self.time_training_start
        prediction : float = self.time_predict_end - self.time_predict_start
        return training, prediction
    
    def get_intercept_and_coef(self):
        return self.theta[0], self.theta[1]
    
    def plot_cost_over_time(self):
        plt.plot(self.costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()

    def plot_predict_x_y(self, X: np.ndarray, y: np.ndarray):
        plt.scatter(X, y)
        plt.plot(X, self.predict(X), color='red')
        plt.show()

