import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from regularization_ import MyRidgeRegression


def high_regression_degree(
        PolynomialFeatures,
        LinearRegression,
        X: np.ndarray,
        y: np.ndarray,
        degrees: list[int],
        test_size: float = 0.3
) -> None:
    # Reshape X to ensure it is 2D
    X = X.reshape(-1, 1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    plt.figure(figsize=(15, 10))
    for i, degree in enumerate(degrees):
        # Transform features to polynomial features of the specified degree
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

        # Train a linear regression model
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_poly, y_train)

        # Predict on full range for smoother curves
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_poly = poly_features.transform(X_range)
        y_range_pred = lin_reg.predict(X_range_poly)

        # Predict for the training and testing sets
        y_train_pred = lin_reg.predict(X_train_poly)
        y_test_pred = lin_reg.predict(X_test_poly)

        # Plot predictions vs actual data
        plt.subplot(2, len(degrees) // 2 + len(degrees) % 2, i + 1)
        plt.scatter(X_train, y_train, label="Train Data", alpha=0.7, color='blue')
        plt.scatter(X_test, y_test, label="Test Data", alpha=0.7, color='green')
        plt.plot(X_range, y_range_pred, label=f"Degree {degree} Fit", color='red')
        plt.title(f"Polynomial Degree: {degree}")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()

    plt.tight_layout()
    plt.suptitle("Model Predictions vs Actual Data Across Degrees", fontsize=16, y=1.02)
    plt.show()


def plot_learning_curves(
        model,
        X: np.ndarray,
        y: np.ndarray
) -> None:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(
            X_train[:m],    
            y_train[:m]
        )

        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(np.sqrt(np.mean((y_train_predict - y_train[:m]) ** 2)))
        val_errors.append(np.sqrt(np.mean((y_val_predict - y_val) ** 2)))

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.sqrt(train_errors),
        "r-+",
        linewidth=2,
        label="train"
    )

    plt.plot(
        np.sqrt(val_errors),
        "b-",
        linewidth=3,
        label="val"
    )

    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title("Learning Curves")
    plt.show()



def plot_ridge():
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    # Test data for predictions
    X_test = np.linspace(0, 2, 100).reshape(-1, 1)

    # Alpha values for regularization
    alphas = [0.1, 1.0, 10.0]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Linear Ridge Regression
    for alpha in alphas:
        model = MyRidgeRegression(alpha=alpha)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        axes[0].plot(X_test, y_pred, label=f"α = {alpha:.1f}")
    
    axes[0].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[0].set_title("Linear Ridge Regression")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # Polynomial Ridge Regression
    poly_degree = 10
    for alpha in alphas:
        # Generate polynomial features
        X_poly = np.hstack([X**i for i in range(1, poly_degree + 1)])
        X_test_poly = np.hstack([X_test**i for i in range(1, poly_degree + 1)])
        
        # Train Ridge Regression with polynomial features
        model = MyRidgeRegression(alpha=alpha)
        model.fit(X_poly, y)
        y_pred_poly = model.predict(X_test_poly)
        
        axes[1].plot(X_test, y_pred_poly, label=f"α = {alpha:.1f}")

    axes[1].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[1].set_title("Polynomial Ridge Regression (Degree 10)")
    axes[1].set_xlabel("X")
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle("Effect of Regularization (α) on Ridge Regression", fontsize=16, y=1.05)
    plt.show()
    


if __name__ == "__main__":
    # np.random.seed(0)
    # X = np.linspace(0, 10, 100)
    # y = 0.5 * X**3 - 2 * X**2 + 3 * X - 1 +  1.3 * np.random.normal(0, 10, 100)

    # degrees = [1, 2, 4, 6, 8, 10, 20, 30, 100, 300]
    # high_regression_degree(PolynomialFeatures, LinearRegression, X, y, degrees)

    # model = LinearRegression()
    # plot_learning_curves(model, X.reshape(-1, 1), y)
    plot_ridge()