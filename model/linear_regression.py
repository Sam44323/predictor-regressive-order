# SINGLE-VARIABLE LINEAR REGRESSION TRAINING IMPLEMENTATION

"""
This model trains to find the best values for w (weight) and b (bias)
so that the linear equation f(x) = wx + b fits the training data as closely as possible.
After training, these learned parameters are used in the predict function
to make accurate predictions on new, unseen data.
The entire training process is focused on tuning w and b for optimal prediction.
"""

import numpy as np
import matplotlib.pyplot as plt

# Price-dataset

# sizes (in 1000 sq ft)
X = np.array([0.8, 1.0, 1.2, 1.5, 1.8])

# Prices (in $1000s)
y = np.array([150, 200, 240, 300, 360])

m = len(X)  # number of training examples

def predict(X, w, b):
    """
    Linear model:
    f(x) = wx + b
    """
    return w * X + b

# Cost-function

def compute_cost(X, y, w, b):
    """
    J(w,b) = (1/2m) * sum((prediction - y)^2)
    """
    predictions = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Gradient-computation

def compute_gradients(X, y, w, b):
    """
    Computes gradients:
    dJ/dw and dJ/db
    """
    predictions = predict(X, w, b)
    errors = predictions - y

    dw = (1 / m) * np.sum(errors * X)
    db = (1 / m) * np.sum(errors)

    return dw, db



# Gradient descent computation

def gradient_descent(X, y, w, b, learning_rate, iterations):
    cost_history = []

    for i in range(iterations):
        
        

        # Compute gradients to find how to adjust w and b to reduce error
        dw, db = compute_gradients(X, y, w, b)

        # Update w and b using gradients (learning step)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Compute and store cost to track-model improvement
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.2f}")

    return w, b, cost_history

# main-function

def main():
    # Hyperparameters
    learning_rate = 0.01
    iterations = 1000

    # Initial parameters
    w = 0.0
    b = 0.0

    # Train the model using gradient descent
    w, b, _ = gradient_descent(X, y, w, b, learning_rate, iterations)


    print(f"Trained parameters: w = {w:.2f}, b = {b:.2f}")

    # --- Benchmarking on fresh test data ---
    # Example fresh data (new house sizes and their actual prices)
    X_test = np.array([1.1, 1.4, 1.7])
    y_test = np.array([220, 280, 340])

    # Predict using trained model
    y_pred = predict(X_test, w, b)
    print("\nTest Data Predictions:")
    for x, actual, pred in zip(X_test, y_test, y_pred):
        print(f"Size: {x}k sq ft | Actual: ${actual}k | Predicted: ${pred:.2f}k")

    # Benchmark: Compute test cost (mean squared error)
    test_cost = compute_cost(X_test, y_test, w, b)
    print(f"Test set cost (MSE): {test_cost:.2f}")

    # Plot cost history and regression line
    plt.scatter(X, y, label="Training Data")
    plt.scatter(X_test, y_test, marker='x', color='red', label="Test Data")
    plt.plot(X, predict(X, w, b), label="Regression Line")
    plt.legend()
    plt.show()

if __name__ == "__main__":    main()

