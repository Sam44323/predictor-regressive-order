
# MULTI-VARIABLE LINEAR REGRESSION TRAINING IMPLEMENTATION


"""
This module implements the training of a multi-variable linear regression model
using gradient descent. Fixes applied:
    1. Feature normalization (z-score) for proper convergence
    2. m derived locally inside functions (no g-dependency)
    3. Safe parameter updates (no in-place mutation)
    4. Increased iterations and adjusted learning rate for full convergence
    5. Proper straight-line plot of predictions vs actuals
"""


# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting
import matplotlib.pyplot as plt


# Dataset with multiple-features
# Features: size (in 1000 sq ft) and number of bedrooms
X = np.array([
    [0.8, 2],   # 0.8 (1000 sq ft), 2 bedrooms
    [0.8, 2],   # 0.8 (1000 sq ft), 2 bedrooms
    [1.0, 3],   # 1.0 (1000 sq ft), 3 bedrooms
    [1.2, 3],   # 1.2 (1000 sq ft), 3 bedrooms
    [1.4, 3],   # 1.2 (1000 sq ft), 3 bedrooms
    [1.5, 4],   # 1.5 (1000 sq ft), 4 bedrooms
    [1.8, 4]    # 1.8 (1000 sq ft), 4 bedrooms
])


# Prices (in $1000s)
y = np.array([
    150,  # Price for first house
    150,  # Price for first house
    200,  # Price for second house
    230,  # Price for second house
    240,  # Price for third house
    300,  # Price for fourth house
    360   # Price for fifth house
])


# Get number of training examples (m) and features (n)
m, n = X.shape  # m = 5, n = 2


# --- Feature Normalization (z-score) ---
# axis = 0 means along the column and 1 means along the row
# Compute mean of each feature (column-wise)
X_mean = X.mean(axis=0)
# Compute standard deviation of each feature (column-wise)
X_std  = X.std(axis=0)
# Normalize the features using z-score normalization
X_norm = (X - X_mean) / X_std



# Predict the function for linear regression
def predict(X, w, b):
    """
    Linear model:
    f(x) = w1*x1 + w2*x2 + ... + wn*xn + b
    X: input features (m, n)
    w: weights (n,)
    b: bias (scalar)
    Returns: predictions (m,)
    """
    # Compute dot product of X and w, then add bias b
    return X.dot(w) + b



# Compute mean squared error cost
def compute_cost(X, y, w, b):
    """
    J(w,b) = (1/2m) * sum((prediction - y)^2)
    X: input features
    y: target values
    w: weights
    b: bias
    Returns: cost (scalar)
    """
    m_local = X.shape[0]  # Number of samples
    predictions = predict(X, w, b)  # Model predictions
    # Compute mean squared error cost
    cost = (1 / (2 * m_local)) * np.sum((predictions - y) ** 2)
    return cost



# Compute gradients for weights and bias
def compute_gradients(X, y, w, b):
    """
    Computes gradients dJ/dw and dJ/db.
    X: input features
    y: target values
    w: weights
    b: bias
    Returns: dw (gradient for weights), db (gradient for bias)
    """
    m_local = X.shape[0]  # Number of samples
    predictions = predict(X, w, b)  # Model predictions
    errors = predictions - y        # Prediction errors

    # Gradient for weights: shape (n,)
    dw = (1 / m_local) * X.T.dot(errors)
    # Gradient for bias: scalar
    db = (1 / m_local) * np.sum(errors)

    return dw, db



# Perform gradient descent optimization
def gradient_descent(X, y, w, b, learning_rate, iterations):
    """
    Performs gradient descent to learn w and b.
    X: input features
    y: target values
    w: initial weights
    b: initial bias
    learning_rate: step size
    iterations: number of iterations
    Returns: learned weights, bias, and cost history
    """
    cost_history = []  # To store cost at each iteration

    for i in range(iterations):
        # Compute gradients for current parameters
        dw, db = compute_gradients(X, y, w, b)

        # Update weights and bias (safe, not in-place)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Compute and record cost
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        # Print cost every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return w, b, cost_history


def main():
    learning_rate = 0.1   # higher LR works well post the normalization
    iterations    = 10000  # enough iterations for full-convergence

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

    # Forward-propagation
    y_pred = predict(X_test_norm, w, b)
    print("\nTest Data Predictions:")
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        print(f"  Sample {i+1}: Actual = ${actual}k | Predicted = ${pred:.2f}k")

    test_cost = compute_cost(X_test_norm, y_test, w, b)
    print(f"\nTest set cost (MSE): {test_cost:.4f}")

    # --- Plot-1: Cost history (should show smooth decay curve) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')

    # --- Plot-2: Predicted vs Actual (should be a straight line) ---
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

    plt.xlabel('Actual ($1000s)')
    plt.ylabel('Predicted ($1000s)')
    plt.title('Predicted vs Actual (should align with red line)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("\nPlot saved to regression_plots.png")


if __name__ == "__main__":
    main()