# Logistic Regression (One-vs-All Multiclass Classification)

import numpy as np
import matplotlib.pyplot as plt

# Create dataset
X = [[30, 35], [50, 55], [70, 65], [80, 75], [60, 60], [90, 95],
    [40, 45], [85, 80], [55, 50], [75, 70]]
# (columns: Exam1 score, Exam2 score)
y = [0, 0, 1, 2, 1, 2, 0, 2, 1, 2] #(0=Low, 1=Medium, 2=High)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Define sigmoid function
def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z
# Define compute_cost function (for logistic regression)
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost
# Define compute_gradient function (for logistic regression)
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


        if i % 500 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost{cost:.4f}")

    return w,b


# Define one_vs_all_train function
# Train one classifier for each class (0, 1, 2)
# Use gradient descent inside a loop for each class
def one_vs_all_train(X, y, num_classes, learning_rate, iterations):
    m, n = X.shape
    all_w = np.zeros((num_classes, n))
    all_b = np.zeros(num_classes)
    for c in range(num_classes):
        y_c = np.where(y == c, 1, 0)
        w = np.zeros(n)
        b = 0
        print(f"\nTraining classifier for class {c}")
        w, b = gradient_descent(X, y_c, w, b, learning_rate ,iterations)
        all_w[c] = w
        all_b[c] = b
    return all_w, all_b
# Define predict function
# Compute probability for each class for a given input
# Return class with highest probability
def predict(X, all_w, all_b):
    num_classes = all_w.shape[0]
    probs = []
    for c in range(num_classes):
        z = np.dot(X, all_w[c]) + all_b[c]
        prob = sigmoid(z)
        probs.append(prob)
    return np.argmax(probs)

# Initialize parameters for all classes
num_classes = 3
learning_rate = 0.001
iterations = 3000
# Train model using one_vs_all_train function
all_w, all_b = one_vs_all_train(X, y, num_classes, learning_rate, iterations)
print(f"\n Learned Weights (all_w) = \n{all_w} \n Learned Biases (all_b)= {all_b}")
# Make prediction for new input: exam scores = [65, 68]
# Use trained models to predict class (0, 1, 2)
new_input = np.array([65, 68])
predicted_class = predict(new_input, all_w, all_b)
print(f"\nPredicted Class for input {new_input}: {predicted_class}")

# Plot data points with different colors for each class
# (Low, Medium, High)
colors = ["green", "orange", "red"]
for c in range(num_classes):
    mask = (y == c)
    plt.scatter(X[mask, 0], X[mask, 1], color=colors[c], label=f"Class {c}")

x1_values = np.array([30, 95])
for c in range(num_classes):
    x2_values = -(all_w[c, 0] * x1_values + all_b[c]) / all_w[c, 1]
    plt.plot(x1_values, x2_values, color=colors[c], label=f"Decision Boundary {c}")

plt.xlabel("Exam Score 1")
plt.ylabel("Exam score 2")
plt.title("One-vs-All Multiclass Classification Data Points")
plt.legend()
plt.grid(True)
plt.show()