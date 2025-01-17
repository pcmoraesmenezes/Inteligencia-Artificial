import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from regularization_ import MyRidgeRegression, MyLassoRegression, MyElasticNetRegression


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
    
def plot_lasso():
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

    # Linear Lasso Regression
    for alpha in alphas:
        model = MyLassoRegression(alpha=alpha)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        axes[0].plot(X_test, y_pred, label=f"α = {alpha:.1f}")
    
    axes[0].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[0].set_title("Linear Lasso Regression")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # Polynomial Lasso Regression
    poly_degree = 10
    for alpha in alphas:
        # Generate polynomial features
        X_poly = np.hstack([X**i for i in range(1, poly_degree + 1)])
        X_test_poly = np.hstack([X_test**i for i in range(1, poly_degree + 1)])
        
        # Train Lasso Regression with polynomial features
        model = MyLassoRegression(alpha=alpha)
        model.fit(X_poly, y)
        y_pred_poly = model.predict(X_test_poly)
        
        axes[1].plot(X_test, y_pred_poly, label=f"α = {alpha:.1f}")

    axes[1].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[1].set_title("Polynomial Lasso Regression (Degree 10)")
    axes[1].set_xlabel("X")
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle("Effect of Regularization (α) on Lasso Regression", fontsize=16, y=1.05)
    plt.show()

def plot_elastic_net():
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

    # Linear Elastic Net Regression
    for alpha in alphas:
        model = MyElasticNetRegression(alpha=alpha, r=0.5)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        axes[0].plot(X_test, y_pred, label=f"α = {alpha:.1f}")
    
    axes[0].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[0].set_title("Linear Elastic Net Regression")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    # Polynomial Elastic Net Regression
    poly_degree = 10
    for alpha in alphas:
        # Generate polynomial features
        X_poly = np.hstack([X**i for i in range(1, poly_degree + 1)])
        X_test_poly = np.hstack([X_test**i for i in range(1, poly_degree + 1)])
        
        # Train Elastic Net Regression with polynomial features
        model = MyElasticNetRegression(alpha=alpha, r=0.5)
        model.fit(X_poly, y)
        y_pred_poly = model.predict(X_test_poly)
        
        axes[1].plot(X_test, y_pred_poly, label=f"α = {alpha:.1f}")

    axes[1].scatter(X, y, color="blue", alpha=0.5, label="Training Data")
    axes[1].set_title("Polynomial Elastic Net Regression (Degree 10)")
    axes[1].set_xlabel("X")
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle("Effect of Regularization (α) on Elastic Net Regression", fontsize=16, y=1.05)
    plt.show()
    

from sklearn.preprocessing import StandardScaler

def compare_regularizations_adjusted_fixed():
    # Generate synthetic data with considerable noise
    np.random.seed(42)
    X = 4 + 20 * np.random.rand(100, 1)
    y = 2 + 3 * X + 0.5 * X**2 + np.random.randn(100, 1)

    # Generate polynomial features of high degree
    poly_degree = 100
    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Standardize features to avoid large coefficients
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_poly_scaled = scaler_X.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler_X.transform(X_test_poly)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Alpha values for regularization
    alphas = [0.1, 1.0, 10.0]

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True, sharey=True)

    # Overfitted Polynomial Regression (Degree 100)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly_scaled, y_train_scaled)
    y_pred_poly_scaled = lin_reg.predict(X_test_poly_scaled)
    y_pred_poly = scaler_y.inverse_transform(y_pred_poly_scaled.reshape(-1, 1))

    axes[0, 0].scatter(X_test, y_test, color="blue", alpha=0.5, label="Test Data")
    axes[0, 0].plot(X_test, y_pred_poly, label="Overfitting (Degree 100)", color="red")
    axes[0, 0].set_title("Overfitted Polynomial Regression")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].legend()

    # Ridge Regression
    for alpha in alphas:
        ridge_model = MyRidgeRegression(alpha=alpha)
        ridge_model.fit(X_train_poly_scaled, y_train_scaled.ravel())
        y_pred_ridge_scaled = ridge_model.predict(X_test_poly_scaled)
        y_pred_ridge = scaler_y.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1))
        axes[0, 1].plot(X_test, y_pred_ridge, label=f"Ridge α={alpha}", linewidth=2)
    
    axes[0, 1].scatter(X_test, y_test, color="blue", alpha=0.5, label="Test Data")
    axes[0, 1].set_title("Ridge Regression")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].legend()

    # Lasso Regression
    for alpha in alphas:
        lasso_model = MyLassoRegression(alpha=alpha)
        lasso_model.fit(X_train_poly_scaled, y_train_scaled.ravel())
        y_pred_lasso_scaled = lasso_model.predict(X_test_poly_scaled)
        y_pred_lasso = scaler_y.inverse_transform(y_pred_lasso_scaled.reshape(-1, 1))
        axes[1, 0].plot(X_test, y_pred_lasso, label=f"Lasso α={alpha}", linewidth=2)

    axes[1, 0].scatter(X_test, y_test, color="blue", alpha=0.5, label="Test Data")
    axes[1, 0].set_title("Lasso Regression")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].legend()

    # Elastic Net Regression
    for alpha in alphas:
        elastic_net_model = MyElasticNetRegression(alpha=alpha, r=0.5)
        elastic_net_model.fit(X_train_poly_scaled, y_train_scaled.ravel())
        y_pred_elastic_scaled = elastic_net_model.predict(X_test_poly_scaled)
        y_pred_elastic = scaler_y.inverse_transform(y_pred_elastic_scaled.reshape(-1, 1))
        axes[1, 1].plot(X_test, y_pred_elastic, label=f"Elastic Net α={alpha}, r=0.5", linewidth=2)

    axes[1, 1].scatter(X_test, y_test, color="blue", alpha=0.5, label="Test Data")
    axes[1, 1].set_title("Elastic Net Regression")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle("Comparison of Regularizations with Test Data", fontsize=18, y=1.02)
    plt.show()


if __name__ == "__main__":
    plot_ridge()
    plot_lasso()
    plot_elastic_net()
    compare_regularizations_adjusted_fixed()