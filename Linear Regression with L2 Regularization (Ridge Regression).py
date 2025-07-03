# Linear Regression with L2 Regularization (Ridge Regression)

import numpy as np
import matplotlib.pyplot as plt

# Create dataset (house size vs price)
X = [500, 600, 700, 850, 1200, 1500, 2000]
y = [120, 150, 200, 250, 350, 450, 600]


# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Compute mean and std of X (for feature scaling)
X_mean = np.mean(X)
X_std = np.std(X)

# Feature scale X (Standardization)
X_scaled = (X - X_mean) / X_std

# Define predict function (linear combination)
def predict(X, w, b):
    z = np.dot(X, w) + b

    return z 

# Define compute_cost function with L2 Regularization
def compute_cost(X, y, w, b, lambda_):
    m = len(y)
    cost = 0
    for i in range(m):
        f_wb_i = predict(X[i], w, b)
        cost += (f_wb_i - y[i])**2
    cost = cost / (2*m)
    reg_term = (lambda_ / (2*m)) * (w**2)
    return  cost + reg_term
    

# Define compute_gradient function with L2 Regularization
# Regularize gradients for w but not for b
def compute_gradient(X, y, w, b, lambda_):
    m = len(y)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_i = predict(X[i], w, b)
        dj_dw += (f_wb_i - y[i]) * X[i]
        dj_db += (f_wb_i - y[i])
    dj_dw = (dj_dw / m) + (lambda_ / m) * w
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
            print(f"Iterations: {i}, Cost: {cost:.4f}")

    return w, b

# Initialize parameters (w as 0, b as 0)
w = 0
b = 0
learning_rate = 0.01
iterations = 5000
lambda_ = 0.1

# Run gradient descent and print final cost and parameters
w, b = gradient_descent(X_scaled, y, w, b, learning_rate, iterations, lambda_)
print(f"Final Paramters: w = {w:.4f}, b = {b:.4f}")
# Plot data points and fitted regression line
x_values = np.linspace(400, 2100, 100)
x_scaled_values = (x_values - X_mean) / X_std
y_values = predict(x_scaled_values, w, b)

plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(x_values, y_values, color="red", label="Regularized Linear Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($1000)")
plt.title("Linear Regression with L2 Regularization")
plt.legend()
plt.grid(True)
plt.show()


# Predict house price for a new size (with scaling)
new_house_size = 1600
new_scaled = (new_house_size - X_mean) / X_std
predicted_price = predict(new_scaled, w, b)
print(f"Predicted price for a {new_house_size} sq ft house: ${predicted_price:.2f}K")

# Compare prediction with and without regularization
# Retrain model without regularization (lambda_ = 0)
lambda_no_reg = 0.0
w_no_reg = 0
b_no_reg = 0

# Run gradient descent again without regularization
w_no_reg, b_no_reg = gradient_descent(X_scaled, y, w_no_reg, b_no_reg, learning_rate, iterations, lambda_no_reg)
print(f"Final Paramters: w = {w_no_reg:.4f}, b = {b_no_reg:.4f}")
# Predict price for same new house size without regularization
predicted_price_no_reg = predict(new_scaled, w_no_reg, b_no_reg)

# Print both predictions
print(f"\nComparison for a {new_house_size} sq ft house:")
print(f"🔸 With Regularization (λ = {lambda_}): ${predicted_price:.2f}K")
print(f"🔸 Without Regularization (λ = 0): ${predicted_price_no_reg:.2f}K")
