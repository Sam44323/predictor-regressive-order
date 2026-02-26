# Multi-Variable Linear Regression: Step-by-Step Execution

This document explains the synchronous execution flow of the multi_linear_regression.py model, from initialization to training and prediction.

---

## 1. Data Preparation
- The dataset is defined:
  - `X`: 2D array of features (e.g., house size and number of bedrooms)
  - `y`: Array of target values (house prices)
  - `m, n`: Number of training examples and features

## 2. Model Definition
- **predict(X, w, b)**
  - Computes predictions using the linear equation: $f(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
  - Uses vectorized dot product for efficiency

## 3. Cost Function
- **compute_cost(X, y, w, b)**
  - Calculates the mean squared error between predictions and actual values

## 4. Gradient Computation
- **compute_gradients(X, y, w, b)**
  - Computes the gradients (partial derivatives) of the cost function with respect to each weight (`w`) and the bias (`b`)
  - Uses vectorized operations for efficiency

## 5. Training: Gradient Descent
- **gradient_descent(X, y, w, b, learning_rate, iterations)**
  - Initializes an empty list to store cost history
  - For each iteration:
    1. Calls `compute_gradients` to get `dw` and `db`
    2. Updates `w` and `b` using the gradients and learning rate
    3. Calls `compute_cost` to calculate and store the current cost
    4. Prints the cost every 100 iterations for monitoring
  - Returns the trained `w`, `b`, and the cost history

## 6. Main Execution
- **main()**
  - Sets hyperparameters: `learning_rate` and `iterations`
  - Initializes `w` (vector of zeros) and `b` (zero)
  - Calls `gradient_descent` to train the model
  - Prints the final trained weights and bias
  - Benchmarks the model on fresh test data:
    - Predicts on new feature samples
    - Prints predicted vs actual values
    - Computes and prints test set cost (MSE)
  - Plots the cost function history for visualization

## 7. Script Entry Point
- If the script is run directly, `main()` is executed

---

## Summary Flow
1. Data is loaded
2. Model and helper functions are defined
3. Training loop (gradient descent) iteratively improves `w` and `b`
4. Final model parameters are used for prediction and benchmarking

This step-by-step process ensures the model learns the best fit for the data and can make predictions on new inputs with multiple features.
