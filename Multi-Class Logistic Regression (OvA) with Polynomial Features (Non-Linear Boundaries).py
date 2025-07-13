# Multi-Class Logistic Regression (OvA) with Polynomial Features (Non-Linear Boundaries)

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)

# Create multi-class dataset with a non-linear pattern
X = np.round(np.random.rand(150, 2), 2)
y = np.array([0 if (x1**2 + x2**2 < 0.5) else 1 if (x1 > 0.5 and x2 > 0.5) else 2 for x1, x2 in X])

# Generate polynomial features
x1 = X[:, 0]
x2 = X[:, 1]
X_poly = np.column_stack((x1, x2, x1**2, x2**2, x1*x2))

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i] * np.log(f_wb_i + 1e-8) - (1 - y[i]) * np.log(1 - f_wb_i + 1e-8)
    cost = cost / m
    return cost

# Compute gradients
def compute_gradients(X, y, w, b):
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

# Gradient descent function
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradients(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db


        if i % 1000 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost = {cost:.2f}")

    return w, b

# Train One-vs-All logistic regression models
def one_vs_all_train(X, y, num_classes, learning_rate, iterations):
    m, n = X.shape
    all_w = np.zeros((num_classes, n))
    all_b = np.zeros(num_classes)
    for c in range(num_classes):
        y_c = np.where(y == c, 1, 0).astype(int)
        w = np.zeros(n)
        b = 0
        print(f"Training Classifier for Class: {c}")
        w, b = gradient_descent(X, y_c, w, b, learning_rate, iterations)
        all_w[c] = w
        all_b[c] = b
    return all_w, all_b

# Prediction function for One-vs-All
def predict_OvA(X, all_w, all_b):
    num_classes = all_w.shape[0]
    probs = []
    for c in range(num_classes):
        z = np.dot(X, all_w[c]) + all_b[c]
        prob = sigmoid(z)
        probs.append(prob)
    probs = np.vstack(probs)
    return np.argmax(probs, axis=0)

# Initialize Parameters
num_classes = 3
learning_rate = 0.3
iterations = 5000

# Train the Model
all_w, all_b = one_vs_all_train(X_poly, y, num_classes, learning_rate, iterations)
print(f"Learned Weights (all_w) = \n{all_w} \nLearned Biases (all_b) = {all_b}")

# Compute accuracy
prediction = predict_OvA(X_poly, all_w, all_b)
accuracy = np.mean(prediction == y) * 100
print(f"Training Accuracy: {accuracy:.2f}")

# Visualize decision boundaries with contourf
x1_values = np.linspace(0, 1, 200)
x2_values = np.linspace(0, 1, 200)
xx, yy = np.meshgrid(x1_values, x2_values)
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
        probs = sigmoid(np.dot(all_w, features)+ all_b)
        zz[i, j] = np.argmax(probs)

plt.contourf(xx, yy, zz, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=['#FFA07A', '#90EE90', "#87CEFA"])
# Scatter actual class points
for c in range(num_classes):
    mask = (y == c)
    plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {c}")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Multi-Class(OvA) with non-linear Boundary")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# Predict new sample
new_input = np.array([0.6, 0.4, 0.6**2, 0.4**2, 0.6*0.4])
predict = predict_OvA(new_input, all_w, all_b)
print(f"\nPredicted Class for new input {new_input}: {predict}")
