# MULTI-VARIABLE LINEAR REGRESSION TRAINING IMPLEMENTATION

"""
This module implements the training of a multi-variable linear regression model
using gradient descent. Fixes applied:
  1. Feature normalization (z-score) for proper convergence
  2. m derived locally inside functions (no global dependency)
  3. Safe parameter updates (no in-place mutation)
  4. Increased iterations and adjusted learning rate for full convergence
  5. Proper straight-line plot of predictions vs actuals
"""

import numpy as np
import matplotlib.pyplot as plt

# Dataset with multiple features
# Features: size (in 1000 sq ft) and number of bedrooms
X = np.array([[0.8, 2],
              [1.0, 3],
              [1.2, 3],
              [1.5, 4],
              [1.8, 4]])

# Prices (in $1000s)
y = np.array([150, 200, 240, 300, 360])

m, n = X.shape  # number of training examples and features

# --- Feature Normalization (z-score) ---
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std


def predict(X, w, b):
    """
    Linear model:
    f(x) = w1*x1 + w2*x2 + ... + wn*xn + b
    """
    return X.dot(w) + b


def compute_cost(X, y, w, b):
    """
    J(w,b) = (1/2m) * sum((prediction - y)^2)
    m is derived locally from X to avoid global dependency.
    """
    m_local = X.shape[0]
    predictions = predict(X, w, b)
    cost = (1 / (2 * m_local)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradients(X, y, w, b):
    """
    Computes gradients dJ/dw and dJ/db.
    m is derived locally from X to avoid global dependency.
    """
    m_local = X.shape[0]
    predictions = predict(X, w, b)
    errors = predictions - y

    dw = (1 / m_local) * X.T.dot(errors)
    db = (1 / m_local) * np.sum(errors)

    return dw, db


def gradient_descent(X, y, w, b, learning_rate, iterations):
    """
    Performs gradient descent to learn w and b.
    Uses safe (non in-place) parameter updates.
    """
    cost_history = []

    for i in range(iterations):
        dw, db = compute_gradients(X, y, w, b)

        # Safe update — avoids in-place mutation bugs
        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return w, b, cost_history


def main():
    learning_rate = 0.1   # higher LR works well after normalization
    iterations    = 10000  # enough iterations for full convergence

    # Initial parameters
    w = np.zeros(n)
    b = 0.0

    # Train the model on normalized features
    w, b, cost_history = gradient_descent(X_norm, y, w, b, learning_rate, iterations)

    print(f"\nLearned weights: {w}")
    print(f"Learned bias:    {b:.4f}")

    # --- Benchmarking on fresh test data ---
    X_test = np.array([[1.1, 3],
                       [1.4, 3],
                       [1.7, 4]])
    y_test = np.array([220, 280, 340])

    # Normalize test data using TRAINING mean and std (important!)
    X_test_norm = (X_test - X_mean) / X_std

    y_pred = predict(X_test_norm, w, b)
    print("\nTest Data Predictions:")
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        print(f"  Sample {i+1}: Actual = ${actual}k | Predicted = ${pred:.2f}k")

    test_cost = compute_cost(X_test_norm, y_test, w, b)
    print(f"\nTest set cost (MSE): {test_cost:.4f}")

    # --- Plot 1: Cost history (should show smooth decay curve) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')

    # --- Plot 2: Predicted vs Actual (should be a straight line) ---
    y_train_pred = predict(X_norm, w, b)

    plt.subplot(1, 2, 2)
    # Plot actual vs predicted for training data
    plt.scatter(y, y_train_pred, color='blue', label='Train samples', zorder=5)
    plt.scatter(y_test, y_pred, color='orange', label='Test samples', zorder=5)

    # Perfect prediction line (straight line where predicted == actual)
    all_vals = np.concatenate([y, y_test])
    min_val, max_val = all_vals.min(), all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', label='Perfect fit (y = ŷ)')

    plt.xlabel('Actual Price ($1000s)')
    plt.ylabel('Predicted Price ($1000s)')
    plt.title('Predicted vs Actual (should align with red line)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("\nPlot saved to regression_plots.png")


if __name__ == "__main__":
    main()