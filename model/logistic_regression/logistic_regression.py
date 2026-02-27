# LOGISTIC REGRESSION TRAINING IMPLEMENTATION

"""
This model trains to find the best values for w (weight) and b (bias)
so that the sigmoid function f(x) = 1 / (1 + e^(-(wx + b))) outputs a
probability between 0 and 1 — representing the chance an input belongs to class 1.

Unlike linear-regression which predicts continuous values, logistic regression
predicts PROBABILITIES and classifies them (>=0.5 → class 1, <0.5 → class 0).

Post the training, the learned w and b are used in predict() to classify new data.
The entire training process is focused on tuning w and b for optimal classification.

NOTE: All ML math (sigmoid, log-loss, gradients) is implemented from scratch.
      matplotlib is used only for visualization — not for any learning logic.
"""

import math
import matplotlib.pyplot as plt

# Student-dataset

# Hours of. studies
X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Passed exam? (1 = yes, 0 = no)
y = [0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ]

m = len(X)  # number of training examples


# Sigmoid-function

def sigmoid(z):
    """
    σ(z) = 1 / (1 + e^(-z))

    WHAT IT DOES:
    Takes any real number z and squashes it into the range (0, 1).
    This output is interpreted as a probability.

    WHY WE NEED IT:
    Linear regression gives raw scores: -∞ to +∞. That's useless for
    classification — you can't say "probability = 347". Sigmoid acts like
    a soft on/off switch. Feed it a big positive number → output near 1
    (confident it's class 1). Feed it a big negative → output near 0
    (confident it's class 0). Feed it 0 → output 0.5 (total uncertainty).

    WHAT HAPPENS WITHOUT IT:
    The model outputs raw numbers with no probabilistic meaning. Log-loss
    cost function breaks. Gradient descent has no clean surface to descend.
    The entire classification framework collapses.
    """
    return 1 / (1 + math.exp(-z))


# Predict-function

def predict(X_input, w, b):
    """
    Logistic model:
    f(x) = σ(wx + b)

    WHAT IT DOES:
    For each input x, computes the linear score z = wx + b,
    then passes it through sigmoid to get a probability in (0, 1).

    WHY WE NEED IT:
    This IS the model. w controls how steep the S-curve is (how sensitive
    the prediction is to hours studied). b shifts the curve left or right
    (adjusts the decision boundary — how many hours is the passing threshold).

    WHAT HAPPENS WITHOUT IT:
    No model = nothing to train, nothing to evaluate, nothing to predict.
    Everything else in this file exists purely to improve this one function.
    """
    return [sigmoid(w * x + b) for x in X_input]


# Cost-function

def compute_cost(X_data, y_data, w, b):
    """
    J(w,b) = -(1/m) * sum[ y*log(p) + (1-y)*log(1-p) ]

    WHERE: p = sigmoid(wx + b) — the predicted probability for each example.

    WHAT IT DOES:
    Measures how wrong the model is across all training examples using
    Binary Cross-Entropy (Log Loss).

    WHY LOG LOSS INSTEAD OF MSE:
    If we used Mean Squared Error with sigmoid, the cost surface becomes
    bumpy and non-convex — full of local minima where gradient descent
    gets stuck. Log loss is perfectly convex (bowl-shaped), so gradient
    descent is mathematically guaranteed to find the global minimum.

    HOW THE PENALTY WORKS:
    - When y=1 (student passed): cost = -log(p).
      If p→1 (correct prediction), cost→0. No penalty.
      If p→0 (wrong prediction),  cost→∞. Massive penalty.
    - When y=0 (student failed): cost = -log(1-p).
      If p→0 (correct), cost→0. No penalty.
      If p→1 (wrong),   cost→∞. Massive penalty.

    Thus the cost SCREAMS when the model is confidently wrong, and
    whispers when it's right. This is exactly the right behavior.

    WHAT HAPPENS WITHOUT IT:
    No measure of "how wrong" means gradient descent has nothing to minimize.
    Training is impossible — like running a race with no finish line.
    """
    predictions = predict(X_data, w, b)
    total_cost = 0

    for i in range(len(X_data)):
        p = predictions[i]

        # Clip p away from 0 and 1 to avoid log(0) = -infinity which crashes training
        p = max(1e-15, min(1 - 1e-15, p))

        # Each example contributes: y*log(p) + (1-y)*log(1-p)
        total_cost += y_data[i] * math.log(p) + (1 - y_data[i]) * math.log(1 - p)

    # Negate because log of a probability (0-1) is always negative,
    # and we want cost to be a positive number we can minimize
    return -total_cost / len(X_data)


# Gradient-computation

def compute_gradients(X_data, y_data, w, b):
    """
    Computes gradients:
    dJ/dw = (1/m) * sum((p - y) * x)
    dJ/db = (1/m) * sum(p - y)

    WHAT IT DOES:
    Calculates the slope of the cost surface with respect to w and b.
    These slopes tell us: "if I nudge w up slightly, does cost go up or down?"

    WHY WE NEED IT:
    Gradient = direction of steepest ASCENT on the cost hill.
    We negate it in gradient descent to go DOWNHILL (reduce cost).
    Without knowing the slope, we'd have no idea which direction
    to-move w and b to improve the model.

    ELEGANT COINCIDENCE:
    The gradient formula (p - y) * x looks identical to linear regression!
    This happens because log-loss and sigmoid cancel each other cleanly
    through the chain rule of calculus. The math works out beautifully.

    WHAT HAPPENS WITHOUT IT:
    Gradient descent has no direction. We'd be guessing how to update
    w and b randomly — never converging to the correct solution.
    """
    predictions = predict(X_data, w, b)   # get all predicted probabilities
    errors = [predictions[i] - y_data[i] for i in range(len(X_data))]  # (p - y) per example

    # dJ/dw: weight each error by its x value, then average
    # x scales the sensitivity — larger x means w has more leverage on that example
    dw = (1 / len(X_data)) * sum(errors[i] * X_data[i] for i in range(len(X_data)))

    # dJ/db: plain average of errors — b shifts all outputs equally so no x weighting
    db = (1 / len(X_data)) * sum(errors)

    return dw, db


# Gradient-descent

def gradient_descent(X_data, y_data, w, b, learning_rate, iterations):
    """
    Iteratively overall updates w and b to minimize cost-function.

    WHAT IT DOES:
    Runs a loop where each iteration:
    1. Computes which direction reduces cost (gradients)
    2. Takes a small step in that direction (update rule)
    3. Records the cost to confirm improvement is happening

    Think of it like being blindfolded on a hilly landscape.
    Each step: feel the slope under your feet (gradient),
    step downhill (update w and b). Repeat until you reach
    the valley floor (minimum cost = best possible w and b).

    WHAT HAPPENS WITHOUT IT:
    w and b stay at 0.0 forever. The model always predicts 0.5
    for every input — completely useless for classification.
    """
    cost_history = []

    for i in range(iterations):

        # Compute gradients — which direction is uphill on the cost-surface?
        dw, db = compute_gradients(X_data, y_data, w, b)

        # Step OPPOSITE to gradient (downhill), scaled by learning_rate
        # Too large a learning_rate → overshoot the valley → cost diverges
        # Too small a learning_rate → tiny steps → training takes forever
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the cost at this iteration to plot learning-curve later
        cost = compute_cost(X_data, y_data, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return w, b, cost_history


# Classify-function

def classify(probabilities, threshold=0.5):
    """
    Converts raw probabilities → hard class labels (0 or 1).

    WHAT IT DOES:
    Applies a decision threshold. p >= 0.5 → predict 1 (pass), else 0 (fail).

    WHY WE NEED IT:
    predict() returns probabilities like 0.73 or 0.21. A final answer
    must be binary: pass or fail. This function makes that hard call.

    WHAT HAPPENS WITHOUT IT:
    You can see probabilities but can't compute accuracy or deliver
    a concrete yes/no classification decision.
    """
    return [1 if p >= threshold else 0 for p in probabilities]


# Accuracy-function

def compute_accuracy(y_true, y_pred_labels):
    """
    Accuracy = correct predictions / total predictions

    WHAT IT DOES:
    Compares predicted labels to actual labels, returns fraction correct.

    WHY WE NEED IT:
    Cost (log-loss) is a mathematical training tool — it's not human-readable.
    Accuracy translates model performance into plain language: "90% correct."
    Both metrics serve different purposes: cost guides training, accuracy
    communicates results.

    WHAT HAPPENS WITHOUT IT:
    You'd have no intuitive way to know if the model is actually good.
    A log-loss of 0.18 means nothing to most people.
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred_labels) if true == pred)
    return correct / len(y_true)


# main-function

def main():
    learning_rate = 0.1   # alpha — step size per gradient descent update
    iterations    = 1000  # training-iterations

    # Initial parameters — start at zero, model has no opinion yet
    w = 0.0
    b = 0.0

    # Train the model using gradient descent
    w, b, cost_history = gradient_descent(X, y, w, b, learning_rate, iterations)

    print(f"\nTrained parameters: w = {w:.4f}, b = {b:.4f}")

    # --- Benchmarking on fresh test data ---
    # New students the model has never seen during training
    X_test = [3.5, 5.5, 7.5]
    y_test = [0,   1,   1  ]

    # Get predicted probabilities from the trained model
    y_prob   = predict(X_test, w, b)

    # Convert probabilities to hard 0/1 class labels
    y_labels = classify(y_prob)

    print("\nTest Data Predictions:")
    for x, actual, prob, label in zip(X_test, y_test, y_prob, y_labels):
        print(f"Hours: {x} | Actual: {actual} | Probability: {prob:.4f} | Predicted: {label}")

    accuracy  = compute_accuracy(y_test, y_labels)
    print(f"\nTest Accuracy : {accuracy * 100:.1f}%")
    print(f"Cost at start : {cost_history[0]:.4f}")
    print(f"Cost at end   : {cost_history[-1]:.4f}")

    # --- Plotting ---

    # Build a smooth x-axis for plotting the sigmoid curve (200 evenly spaced points)
    x_range   = [0.5 + i * (10.5 - 0.5) / 200 for i in range(201)]
    sigmoid_curve = predict(x_range, w, b)   # predicted probability at each point

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: Sigmoid decision curve ---
    ax1.plot(x_range, sigmoid_curve, color='steelblue', linewidth=2, label="Sigmoid Curve (Model)")

    # Decision boundary — vertical line at the point where p = 0.5
    # At boundary: wx + b = 0 → x = -b/w
    boundary = -b / w
    ax1.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.2, label=f"Decision Boundary ({boundary:.2f}h)")
    ax1.axhline(y=0.5,      color='gray', linestyle=':',  linewidth=1.0)

    # Training data — plot actual labels as scatter points
    ax1.scatter([X[i] for i in range(m) if y[i] == 1],
                [y[i] for i in range(m) if y[i] == 1],
                color='green', zorder=5, label="Train: Passed (1)")
    ax1.scatter([X[i] for i in range(m) if y[i] == 0],
                [y[i] for i in range(m) if y[i] == 0],
                color='red',   zorder=5, label="Train: Failed (0)")

    # Test data — plot with marker 'x' to distinguish from training
    ax1.scatter([X_test[i] for i in range(len(X_test)) if y_test[i] == 1],
                [y_test[i]  for i in range(len(X_test)) if y_test[i] == 1],
                color='green', marker='x', s=100, linewidths=2, label="Test: Passed (1)")
    ax1.scatter([X_test[i] for i in range(len(X_test)) if y_test[i] == 0],
                [y_test[i]  for i in range(len(X_test)) if y_test[i] == 0],
                color='red',   marker='x', s=100, linewidths=2, label="Test: Failed (0)")

    ax1.set_xlabel("Hours Studied")
    ax1.set_ylabel("Probability of Passing")
    ax1.set_title("Logistic Regression — Sigmoid Curve")
    ax1.legend(fontsize=7.5)
    ax1.set_ylim(-0.1, 1.1)

    # --- Plot 2: Cost history (learning curve) ---
    ax2.plot(range(iterations), cost_history, color='darkorange', linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Log-Loss Cost")
    ax2.set_title("Cost History — Gradient Descent Convergence")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()