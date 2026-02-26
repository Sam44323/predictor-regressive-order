# Linear Regression Model: Step-by-Step Execution

This document explains the synchronous execution flow of the linear_regression.py model, from initialization to training and prediction.

---

## 1. Data Preparation
- The dataset is defined:
  - `X`: Array of sizes (in 1000 sq ft)
  - `y`: Array of prices (in $1000s)
  - `m`: Number of training examples

## 2. Model Definition
- **predict(X, w, b)**
  - Computes predictions using the linear equation: $f(x) = wx + b$

## 3. Cost Function
- **compute_cost(X, y, w, b)**
  - Calculates the mean squared error between predictions and actual values.

## 4. Gradient Computation
- **compute_gradients(X, y, w, b)**
  - Computes the gradients (partial derivatives) of the cost function with respect to `w` and `b`.

## 5. Training: Gradient Descent
- **gradient_descent(X, y, w, b, learning_rate, iterations)**
  - Initializes an empty list to store cost history.
  - For each iteration:
    1. Calls `compute_gradients` to get `dw` and `db`.
    2. Updates `w` and `b` using the gradients and learning rate.
    3. Calls `compute_cost` to calculate and store the current cost.
    4. Prints the cost every 100 iterations for monitoring.
  - Returns the trained `w`, `b`, and the cost history.

## 6. Main Execution
- **main()**
  - Sets hyperparameters: `learning_rate` and `iterations`.
  - Initializes `w` and `b` to zero.
  - Calls `gradient_descent` to train the model.
  - Prints the final trained parameters.
  - Plots the original data and the regression line for visualization.

## 7. Script Entry Point
- If the script is run directly, `main()` is executed.

---

## Summary Flow
1. Data is loaded.
2. Model and helper functions are defined.
3. Training loop (gradient descent) iteratively improves `w` and `b`.
4. Final model parameters are used for prediction and visualization.

This step-by-step process ensures the model learns the best fit for the data and can make predictions on new inputs.