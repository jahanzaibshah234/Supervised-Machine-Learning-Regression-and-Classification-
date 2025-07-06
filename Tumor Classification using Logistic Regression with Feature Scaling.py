# Tumor Classification using Logistic Regression with Feature Scaling

import numpy as np
import matplotlib.pyplot as plt

# Create dataset (e.g., Clump thickness vs Tumor size and result)
X = [
    [3, 2],
    [4, 3],
    [6, 5],
    [7, 6],
    [8, 7],
    [10, 9],
    [2, 1],
    [9, 8]
]
y = [0, 0, 1, 1, 1, 1, 0, 1]

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Compute mean and std of X (for feature scaling)
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

# Feature scale X (Standardization)
x_scaled = (X - x_mean) / x_std

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Define compute_cost function (Binary Cross-Entropy)
def compute_cost(X,y, w, b):
    m = len(y)
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost
# Define compute_gradient function
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        dj_dw += (f_wb_i - y[i]) * X[i]
        dj_db += (f_wb_i - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
# Implement gradient_descent function (update w and b)
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 500 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost {cost:.2f}")

    return w, b

# TODO 8: Initialize parameters (w as zeros, b as 0)
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.01
iterations = 5000


# TODO 9: Run gradient descent and print final cost and parameters
w, b = gradient_descent(x_scaled, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w}, b = {b:.4f}")

# Plot decision boundary and data points (scatter plot for both classes)
x1_values = np.linspace(-2, 2, 100)
x2_values = (-w[0] * x1_values + b) / w[1]

plt.scatter(x_scaled[y==0, 0], x_scaled[y==0, 1], color="green", label="Benign (0)")
plt.scatter(x_scaled[y==1, 0], x_scaled[y==1, 1], color="red", label="Malignant (1)")
plt.plot(x1_values, x2_values, color="blue", label="Decision Boundary")

plt.xlabel("Clump Thickness (scaled)")
plt.ylabel("Tumor Size (scaled)")
plt.title("Tumor Classification using Logistic Regression with Feature Scaling")
plt.legend()
plt.grid(True)
plt.show()
# Make prediction for a new tumor and print probability and predicted class
new_tumor = np.array([4, 6])
new_scaled = (new_tumor - x_mean) / x_std
probability = sigmoid(np.dot(new_scaled, w) + b)
prediction = 1 if probability >= 0.5 else 0

print(f"Predicted Probability: {probability:.4f}")
print(f"Predicted Class: {'Malignant (1)' if prediction == 1 else 'Benign (0)'}")