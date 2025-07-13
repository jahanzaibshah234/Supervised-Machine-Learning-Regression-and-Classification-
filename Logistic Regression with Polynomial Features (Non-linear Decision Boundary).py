# Logistic Regression with Polynomial Features (Non-linear Decision Boundary)

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Create 2D dataset (100 points) between 0 and 1
X = np.round(np.random.rand(100, 2), 2)

# Create labels using a nonlinear rule (like XOR)
y = np.array([1 if (x1 > 0.5 and x2 < 0.5) or (x1 < 0.5 and x2 > 0.5) else 0 for x1, x2 in X])

# Generate polynomial features (x1, x2, x1^2, x2^2, x1*x2)
x1 = X[:, 0]
x2 = X[:, 1]
X_poly = np.column_stack((x1, x2, x1**2, x2**2, x1*x2))

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost (Binary Cross-Entropy Loss)
def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i] * np.log(f_wb_i + 1e-8) - (1 - y[i]) * np.log(1 - f_wb_i + 1e-8)
    return cost / m

# Compute gradients
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

# Gradient Descent
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        if i % 1000 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return w, b

# Initialize parameters
m, n = X_poly.shape
w = np.zeros(n)
b = 0
learning_rate = 0.5
iterations = 10000

# Train the model
w, b = gradient_descent(X_poly, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w}, b = {b:.4f}")

# Prediction function
def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= 0.5).astype(int)

# Calculate training accuracy
prediction = predict(X_poly, w, b)
accuracy = np.mean(prediction == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")

# Visualize decision boundary
x1_vals = np.linspace(0, 1, 200)
x2_vals = np.linspace(0, 1, 200)
xx, yy = np.meshgrid(x1_vals, x2_vals)
zz = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        features = np.array([
            xx[i, j],
            yy[i, j],
            xx[i, j]**2,
            yy[i, j]**2,
            xx[i, j]*yy[i, j]
        ])
        zz[i, j] = sigmoid(np.dot(features, w) + b)

plt.contour(xx, yy, zz, levels=[0.5], colors='black')

# Scatter plot of data points
for label in [0, 1]:
    mask = (y == label)
    plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {label}")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Logistic Regression with Polynomial Features (Non-linear Decision Boundary)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()