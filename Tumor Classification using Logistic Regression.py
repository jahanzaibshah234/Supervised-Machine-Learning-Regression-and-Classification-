# Project: Tumor Classification using Logistic Regression

import numpy as np
import matplotlib.pyplot as plt

# Create data
X = [1, 2, 3, 4, 5, 6, 7] #(tumor sizes in cm)
y = [0, 0, 0, 1, 1, 1, 1] #(0 = benign, 1 = malignant)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Define sigmoid function
def sigmoid(z):
    z = 1 / (1 + np.exp(-z))

    return z


# Define compute_cost function (log loss)
def compute_cost(X, y, w, b):
    m = len(X)
    cost = 0
    for i in range(m):
        z = w * X[i] + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost
# Define compute_gradient function (for w and b)
def compute_gradient(X, y, w, b):
    m = len(X)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        z = w * X[i] + b
        f_wb_i = sigmoid(z)
        dj_dw += (f_wb_i - y[i]) * X[i]
        dj_db += (f_wb_i - y[i])
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db
# Implement gradient_descent function
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: cost{cost:.4f}")

    return w, b
    

# Initialize parameters w and b
w = 0
b = 0
learning_rate = 0.1
iterations = 1000

# Run gradient descent and print final cost and parameters
w, b = gradient_descent(X, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w:.4f}, b = {b:.4f}")

# Plot decision boundary
# Plot data points: benign (green), malignant (red)
# Plot sigmoid curve
x_values = np.linspace(0, 8, 100)
y_values = sigmoid(w * x_values + b)
plt.scatter(X[y==0], y[y==0], color="green", label="Benign (0)")
plt.scatter(X[y==1], y[y==1], color="red", label="Malignant (1)")
plt.plot(x_values, y_values, color="blue", label="Sigmoid Curve")
plt.xlabel("Tumor Size (cm)")
plt.ylabel("Probability")
plt.title("Logistic Regression - Tumor Classification")
plt.legend()
plt.grid(True)
plt.show()
# Make prediction for a tumor size of 4.5 cm
# Use sigmoid(w*X + b)
# If value > 0.5 → malignant (1), else benign (0)

tumor_size = 4.5
probability = sigmoid(w * tumor_size + b)
prediction = 1 if probability >= 0.5 else 0

print(f"\nPredicted Probability for tumor size: {tumor_size} cm: {probability:.4f}")
print(f"Predicted Class: {'Malignant (1)' if prediction == 1 else 'Benign (0)'}")



