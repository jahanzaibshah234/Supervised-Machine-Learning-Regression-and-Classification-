# Project: Predict Exam Scores based on Study Hours using Linear Regression

import numpy as np
import matplotlib.pyplot as plt

# Create the data
X = [2, 3, 4, 5, 6, 7, 8]
y = [50, 55, 60, 65, 70, 75, 80]

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Define function to compute cost (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = len(X)
    cost = 0
    for i in range(m):
        cost += (w * X[i] + b - y[i])**2
    cost = cost / (2 * m)
    return cost


# Define function to compute gradients (for w and b)
def compute_gradient(X, y, w, b):
    m = len(X)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (w * X[i] + b - y[i]) * X[i]
        dj_db += (w * X[i] + b - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


# Implement Gradient Descent function
def gradient_descent(X, y, w, b, learning_rate, iterations):
    
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)


        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db


        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i}: Cost {cost:.2f}")

    return w, b

# Initialize parameters and run gradient descent
w = 0
b = 0.0
learning_rate = 0.01
iterations = 1000

w, b = gradient_descent(X, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w:.2f}, b = {b:.2f}")


# Plot the regression line with the data points
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X, w * X + b, color='blue', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()

# Make a prediction for a student who studied 9 hours
hours = 9
predicted_score = w * hours + b
print(f"Predicted Score for {hours} hours of study = {predicted_score:.2f}")