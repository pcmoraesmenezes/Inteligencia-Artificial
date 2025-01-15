import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from linear_regression import MyLinearRegression
from gradient_descent_models import GradientDescent, MiniBatchGradientDescent, StoachasticGradientDescent


if __name__ == "__main__":

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    f_x = 4 + 3 * X  

    models = {
        "Linear Regression": MyLinearRegression(),
        "Gradient Descent": GradientDescent(learning_rate=0.1, epochs=10000),
        "Mini Batch GD": MiniBatchGradientDescent(learning_rate=0.1, epochs=10000, batch_size=10),
        "Stochastic GD": StoachasticGradientDescent(learning_rate=0.1, epochs=10000)
    }

    times = {"Training": [], "Prediction": []}
    intercepts = []
    coefficients = []

    print("Exibindo os resultados do modelo (Intercepto e Coeficiente vs Função Real)...")
    fig, axs = plt.subplots(1, len(models), figsize=(16, 4))
    for i, (name, model) in enumerate(models.items()):
        model.fit(X, y)
        predictions = model.predict(X)
        training_time, prediction_time = model.get_times()
        intercept, coef = model.get_intercept_and_coef()
        
        times["Training"].append(training_time)
        times["Prediction"].append(prediction_time)
        intercepts.append(intercept)
        coefficients.append(coef)

        axs[i].scatter(X, y, label="Data")
        axs[i].plot(X, predictions, color='red', label=f"Predicted (Intercept: {intercept.item():.2f}, Coef: {coef.item():.2f})")
        axs[i].plot(X, f_x, color='green', linestyle='dashed', label="True Function")
        axs[i].set_title(name)
        axs[i].legend()
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("y")
    plt.tight_layout()
    plt.show()

    print("Exibindo a função de custo ao longo das épocas...")
    fig, axs = plt.subplots(1, len(models), figsize=(16, 4))
    for i, (name, model) in enumerate(models.items()):
        if name == "Linear Regression":
            costs = model.simulate_cost_over_time(X, y, epochs=100)
            axs[i].plot(costs)
        elif hasattr(model, 'costs') and model.costs:
            axs[i].plot(model.costs)
        axs[i].set_title(f"{name} - Cost Over Time")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Cost")
    plt.tight_layout()
    plt.show()

    print("Exibindo os tempos de treinamento e previsão...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].bar(models.keys(), times["Training"], color='blue', label="Training Time")
    axs[0].set_title("Training Time")
    axs[0].set_ylabel("Time (s)")
    axs[0].legend()

    axs[1].bar(models.keys(), times["Prediction"], color='orange', label="Prediction Time")
    axs[1].set_title("Prediction Time")
    axs[1].set_ylabel("Time (s)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    print("Exibindo os interceptos e coeficientes estimados...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].bar(models.keys(), [inter.item() for inter in intercepts], color='purple', label="Intercepts")
    axs[0].set_title("Intercepts")
    axs[0].set_ylabel("Intercept Value")
    axs[0].legend()

    axs[1].bar(models.keys(), [coef.item() for coef in coefficients], color='green', label="Coefficients")
    axs[1].set_title("Coefficients")
    axs[1].set_ylabel("Coefficient Value")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
