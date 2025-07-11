# Logistic Regression (Multi-class One-vs-All with Contour Decision Boundaries)

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Set Random Seed (for reproducibility)
np.random.seed(0)

# Create Dataset (Multi-class)
X = np.round(np.random.rand(150, 2), 2)
y = np.array([0 if (x1 < 0.5 and x2 < 0.5) else 1 if (x1 >= 0.5 and x2 < 0.5) else 2 for x1, x2 in X]) 

# Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute Cost (Binary Cross-Entropy)
def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i] * np.log(f_wb_i + 1e-8) - (1 - y[i]) * np.log(1 - f_wb_i + 1e-8)
    cost = cost / m
    return cost

# Compute Gradients
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

# Gradient Descent Function
def gradient_descent(X, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 1000 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations {i}: Cost = {cost:.2f}")
    return w, b

# Train One-vs-All Models
# For each class (0, 1, 2) train a separate logistic regression model
# Loop over all classes, convert multi-class labels to binary labels for each one
def one_vs_all_train(X, y, num_classes, learning_rate, iterations):
    m, n = X.shape
    all_w = np.zeros((num_classes, n))
    all_b = np.zeros(num_classes)
    for c in range(num_classes):
        y_c = np.where(y == c, 1, 0).astype(int)
        w = np.zeros(n)
        b = 0
        print(f"\nTraining Classifier for Class: {c}")
        w, b = gradient_descent(X, y_c, w, b, learning_rate, iterations)
        all_w[c] = w
        all_b[c] = b
    return all_w, all_b
        
# Predict Class for Each Sample
def predict_one_vs_all(X, all_w, all_b):
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
learning_rate = 0.1
iterations = 5000

# Train the Model
all_w, all_b = one_vs_all_train(X, y, num_classes, learning_rate, iterations)
print(f"\nLearned Weights (all_w) = \n{all_w} \nLearned Biases (all_b)= {all_b}")

# Compute Accuracy
prediction = predict_one_vs_all(X, all_w, all_b)
accuracy = np.mean(prediction == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}")

# Visualize Decision Boundaries
# use meshgrid, sigmoid, and contour to visualize decision regions for each class
x1_values = np.linspace(0, 1, 200)
x2_values = np.linspace(0, 1, 200)
xx, yy = np.meshgrid(x1_values, x2_values)
zz = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        features = np.array([xx[i, j], yy[i, j]])
        probs = sigmoid(np.dot(all_w, features) + all_b)
        zz[i, j] = np.argmax(probs)

plt.contourf(xx, yy, zz, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], colors=['#FFA07A','#90EE90','#87CEFA'])

# scatter plot colored by actual class labels
for c in range(num_classes):
    mask = (y == c)
    plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {c}")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Multi-Class Logistic Regression (One-vs-all)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# Make prediction for new input
new_input = np.array([0.6, 0.7])
predicted_class = predict_one_vs_all(new_input, all_w, all_b)
print(f"\nPredicted Class for new input {new_input}: {predicted_class}")
