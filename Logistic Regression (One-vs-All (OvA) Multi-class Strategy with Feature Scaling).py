# Logistic Regression (One-vs-All (OvA) Multi-class Strategy with Feature Scaling)

import numpy as np
import matplotlib.pyplot as plt

# Create multi-class dataset & Convert to numpy arrays(e.g., exam scores vs class labels 0, 1, 2)
# (Columns: Exam 1 Score, Exam 2 Score)
X = np.array([
    [30, 34],
    [40, 45],
    [46, 48],
    [50, 56],
    [70, 80],
    [75, 79],
    [66, 68],
    [81, 85],
    [60, 65],
    [90, 94]
])
y = np.array([0, 0, 0, 1, 2, 2, 1, 2, 1, 2])#(0=Low, 1=Medium, 2=High)

# Feature scale X (Standardization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_scaled = (X - X_mean) / X_std

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define compute_cost function (Binary Cross-Entropy for one-vs-all)
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost

# Define compute_gradient function (for one-vs-all)
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
# Implement gradient_descent function
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db


        if i % 500 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost = {cost:.2f}")
        

    return w, b
# Implement one_vs_all_training function (train separate classifier for each class)
def one_vs_all_train(X, y, num_classes, learning_rate, iterations):
    m, n = X.shape
    all_w = np.zeros((num_classes, n))
    all_b = np.zeros(num_classes)
    for c in range(num_classes):
        y_c = np.where(y == c, 1, 0)
        w = np.zeros(n)
        b = 0
        print(f"\nTraining Classifier for Class: {c}")
        w, b = gradient_descent(X, y_c, w, b, learning_rate, iterations)
        all_w[c] = w
        all_b[c] = b
    return all_w, all_b
# Implement one_vs_all_prediction function (pick class with highest probability)
def OvA_predict(X, all_w, all_b):
    num_classes = all_w.shape[0]
    probs = []
    for c in range(num_classes):
        z = np.dot(X, all_w[c]) + all_b[c]
        prob = sigmoid(z)
        probs.append(prob)
    return np.argmax(probs)
# Initialize parameters for all classes
num_classes = 3
learning_rate = 0.01
iterations = 3000

# Train model using one_vs_all_training, print parameters
all_w, all_b = one_vs_all_train(X_scaled, y, num_classes, learning_rate, iterations)
print(f"\n Learned Weights (all_w) = \n{all_w} \n Learned Biases (all_b)= {all_b}")

# Make predictions on training set and print accuracy
prediction = []
for i in range(len(X_scaled)):
    predicted_class = OvA_predict(X_scaled[i], all_w, all_b)
    prediction.append(predicted_class)

predictions = np.array(prediction)
accuracy = np.mean(predictions == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}")

# Make prediction for new input
new_input = np.array([34, 39])
new_scaled = (new_input - X_mean) / X_std
predicted_class = OvA_predict(new_scaled, all_w, all_b)
print(f"\nPredicted Class for input {new_input}: {predicted_class}")

# Plot data points colored by class
colors = ["green", "orange", "red"]

for c in range(num_classes):
    mask = (y == c)
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], color=colors[c], label=f"Class {c}")


x1_values = np.linspace(-2, 2, 100)
for c in range(num_classes):
    x2_values = (-all_w[c, 0] * x1_values + all_b[c]) / all_w[c, 1]
    plt.plot(x1_values, x2_values, color=colors[c], label=f"Boundary Class {c}")

plt.xlabel("Exam 1 Score (scaled)")
plt.ylabel("Exam 2 Score (scaled)")
plt.title("OvA Multiclass Logistic Regression with Feature Scaling")
plt.legend()
plt.grid(True)
plt.show()