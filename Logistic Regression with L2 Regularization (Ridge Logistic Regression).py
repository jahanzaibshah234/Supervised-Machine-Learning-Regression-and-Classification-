# Logistic Regression with L2 Regularization (Ridge Logistic Regression)

import numpy as np
import matplotlib.pyplot as plt

# Create dataset (e.g., exam scores vs admission result)
X = [
    [50, 60],
    [55, 62],
    [61, 65],
    [65, 70],
    [70, 72],
    [75, 80],
    [80, 85],
    [85, 87]
]
y = [0, 0 , 0, 1, 1, 1, 1, 1]

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Compute mean and std of X (for feature scaling)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Feature scale X (Standardization)
X_scaled = (X - X_mean) / X_std

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define compute_cost function (with L2 Regularization term)
def compute_cost(X, y, w, b, lambda_):
    m = len(y)
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    reg_term = (lambda_ / 2*m) * np.sum(w**2)
    return cost + reg_term

# Define compute_gradient function (with L2 Regularization for w, not for b)
def compute_gradient(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        dj_dw += (f_wb_i - y[i]) * X[i]
        dj_db += (f_wb_i - y[i])
    dj_dw = (dj_dw /m) + (lambda_ / m) * w
    dj_db /= m
    return dj_dw, dj_db
# Implement gradient_descent function (update w and b)
def gradient_descent(X, y, w, b, learning_rate, iterations, lambda_):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 500 == 0:
            cost = compute_cost(X, y, w, b, lambda_)
            print(f"Iterations: {i}: Cost {cost:.4f}")

    return w, b
# Initialize parameters (w as zeros, b as 0)
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.01
iterations = 5000
lambda_ = 0.1

# Run gradient descent and print final cost and parameters
w, b = gradient_descent(X_scaled, y, w, b, learning_rate, iterations, lambda_)
print(f"Final Parameters: w = {w}, b = {b:.4f}")

# Plot decision boundary and data points (scatter plot with 2 classes)
x1_values = np.linspace(-2, 2, 100)
x2_values = (-w[0] * x1_values + b) / w[1]

plt.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], color="green", label="Not Admitted (0)")
plt.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], color="red", label="Admitted (1)")
plt.plot(x1_values, x2_values, color="blue", label="Decision Boundary")

plt.xlabel("Exam 1 Score (Scaled)")
plt.ylabel("Exam 2 Score (Scaled)")
plt.title("Logistic Regression with L2 Regularization")
plt.legend()
plt.grid(True)
plt.show()

# Make prediction for a new input and print probability and class
new_input = np.array([81, 83])
new_scaled = (new_input - X_mean) / X_std
probability = sigmoid(np.dot(new_scaled, w) + b)
prediction = 1 if probability >= 0.5 else 0

print(f"Predicted Probability: {probability:.4f}")
print(f"Predicted Class: {'Admitted (1)' if prediction == 1 else 'Not Admitted (0)'}")

