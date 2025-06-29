# Tumor Classification using Logistic Regression with 2 Features (2D Decision Boundary)
# Predict whether a tumor is benign (0) or malignant (1) based on both:
# tumor size (in cm)
# patient’s age (in years)

import numpy as np
import matplotlib.pyplot as plt

# Create dataset
# (columns: tumor size, age)
X = [[1, 45], [2, 50], [3, 54], [4, 55], [5, 60], [6, 65], [7, 70]] 
y = [0, 0, 0, 1, 1, 1, 1] # (0=Benign, 1=Malignant)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Define sigmoid function:
def sigmoid(z):

    return 1 / (1 + np.exp(-z))

# Define compute_cost function (for 2 variables)
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost / m

    return cost

# Define compute_gradient function (for w vector and b)
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

# Implement gradient_descent function (update both w[0] and w[1])
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost:{cost:.4f}")

    return w, b

# Initialize parameters (w as numpy array with two zeros, b=0)
w = np.zeros(2)
b = 0
learning_rate = 0.001
iterations = 3000

# Run gradient descent and print final cost and parameters
w, b = gradient_descent(X, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w}, b = {b:.4f}")


# Plot decision boundary:
# Scatter plot Benign vs Malignant points (different colors)
# Draw a decision boundary line where probability is 0.5
Benign = y == 0
Malignant = y == 1
plt.scatter(X[Benign, 0], X[Benign, 1], color="green", label="Benign (0)")
plt.scatter(X[Malignant, 0], X[Malignant, 1], color="red", label="Malignant (1)")

x1_values = np.array([0, 8])
x2_values = -(w[0] * x1_values + b) / w[1]

plt.plot(x1_values, x2_values, color="blue", label="Decision Boundary")
plt.xlabel("Tumor Size (cm)")
plt.ylabel("Age (Years)")
plt.title("Logistic Regression - Tumor Classification (2 Features)")
plt.legend()
plt.grid(True)
plt.show()


# Make prediction for tumor size=4.5 and age=58
# Use sigmoid(w1*x1 + w2*x2 + b)
# If probability >= 0.5 → Malignant, else Benign
input_features = np.array([4.5, 58])
probability = sigmoid(np.dot(input_features, w) + b)
prediction = 1 if probability >= 0.5 else 0

print(f"Predicted Probability for Tumor Size 4.5 cm and Age 58 years: {probability:.4f} ")
print(f"Predicted Class: {'Malignant (1)' if prediction == 1 else 'Benign (0)'}")

