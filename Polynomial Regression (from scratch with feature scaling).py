# Polynomial Regression (from scratch with feature scaling)
# Predict house price based on house size (square feet) using Polynomial Regression


import numpy as np
import matplotlib.pyplot as plt

# Create dataset
X = [500, 750, 1000, 1250, 1500, 1750, 2000]
y = [120, 150, 200, 220, 260, 275, 310]


# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Compute mean and std of X (for feature scaling)
X_mean = np.mean(X)
X_std = np.std(X)

# Feature scaling (Standardize X)
X_scaled = (X - X_mean) / X_std

# Convert X to polynomial features (add X^2 column)
X_poly = np.column_stack((X_scaled, X_scaled**2))

# Initialize parameters w (for both X and X^2) and b
w = np.zeros(2)
b = 0
# Define predict function (linear combination)
# z = w1 * x1 + w2 * x2 + b
def predict(X, w, b):
    z = np.dot(X, w) + b

    return z
# Define compute_cost function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0
    for i in range(m):
        f_wb_i = predict(X[i], w, b)
        cost += (f_wb_i - y[i])**2 
    return cost / (2*m)

# Define compute_gradient function (for w and b)
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_wb_i = predict(X[i], w, b)
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
            print(f"Iterations {i}: Cost:{cost:.4f}")

    return w, b
# Set learning rate and number of iterations
learning_rate = 0.01
iterations = 5000

# Run gradient descent and print final cost and parameters
w, b = gradient_descent(X_poly, y, w, b, learning_rate, iterations)
print(f"Final Parameters: w = {w}, b = {b:.4f}")

# Predict new house price (example: 1600 sq ft)
new_house_size = 1600
new_scaled_value = (new_house_size - X_mean) / X_std
new_poly = np.array([new_scaled_value, new_scaled_value**2])
predicted_price = predict(new_poly, w, b)
print(f"Predicted price for a {new_house_size} sq ft house: ${predicted_price:.2f}K")


# Plot data points and fitted regression curve
# Use linspace to create smooth curve values
# Convert linspace values to polynomial features too
x_values = np.linspace(400, 2100, 100)
x_scaled_values = (x_values - X_mean) / X_std
x_poly_values = np.column_stack((x_scaled_values, x_scaled_values**2))
y_values = predict(x_poly_values, w, b)

plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(x_values, y_values, color="red", label="Polynomial Regression Curve")
plt.xlabel("House size (sq ft)")
plt.ylabel("House Price in ($1000)")
plt.title("Predict House Price Using Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()
