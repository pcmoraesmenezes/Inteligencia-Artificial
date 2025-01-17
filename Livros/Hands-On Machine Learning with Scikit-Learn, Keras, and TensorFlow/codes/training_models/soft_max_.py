import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None  

    @staticmethod
    def softmax(scores):
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Estabilidade numérica
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _compute_cost(self, X, y_one_hot):

        m = X.shape[0]
        scores = X @ self.theta
        probabilities = self.softmax(scores)
        log_probs = -np.log(probabilities + 1e-9)
        cost = (1 / m) * np.sum(y_one_hot * log_probs)
        return cost

    def fit(self, X, y):

        m, n = X.shape
        K = len(np.unique(y)) 
        self.theta = np.zeros((n, K)) 

        y_one_hot = np.zeros((m, K))
        y_one_hot[np.arange(m), y] = 1

        for _ in range(self.num_iterations):
            scores = X @ self.theta
            probabilities = self.softmax(scores)  

            gradient = (1 / m) * X.T @ (probabilities - y_one_hot) 

            self.theta -= self.learning_rate * gradient

    def predict_proba(self, X):
        scores = X @ self.theta
        return self.softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


    def frontier_plot(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
        
        handles, _ = scatter.legend_elements()
        class_labels = ['Setosa', 'Versicolour', 'Virginica']
        plt.legend(handles, class_labels, loc='upper left')
        
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.title('Softmax Regression Decision Boundary')
        plt.show()

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, (2, 3)]
    y = iris.target
    model = SoftmaxRegression()
    model.fit(X, y)
    model.frontier_plot(X, y)