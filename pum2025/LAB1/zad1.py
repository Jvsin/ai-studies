import numpy as np
from sklearn.datasets import make_regression
from scipy.optimize import minimize
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=16, random_state=95)
X = X.flatten()
# print(X)

def analytical_lr(X, y):
    X_bias = np.vstack([X, np.ones(len(X))]).T ## dodajemy jedynki (wyrazy wolne)
    # print(X_bias)

    w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    # print(w)
    return w[0], w[1]  # a, b (slope / intercept)

def mse(w, X, y):
    X_bias = np.vstack([X, np.ones(len(X))]).T
    results = X_bias @ w
    
    return np.sum((y - results) ** 2) / len(y)

def numerical_lr(X, y):
    w0 = np.zeros(2)
    result = minimize(mse, w0, args=(X, y), method='Powell')
    return result.x[0], result.x[1]

def predict(X, slope, intercept):
    return slope * X + intercept

a1, b1 = analytical_lr(X, y)
a2, b2 = numerical_lr(X, y)

alr_predictions = predict(X, a1, b1)
nlr_predictions = predict(X, a2, b2)

mse_analytical = np.mean((y - alr_predictions) ** 2)
mse_numerical = np.mean((y - nlr_predictions) ** 2)

print(mse_analytical, mse_analytical)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Dane')
plt.plot(X, alr_predictions, color='red', label='Analityczna')
plt.plot(X, nlr_predictions, color='green', linestyle='--', label='Numeryczna')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Porównanie analitycznej i numerycznej regresji liniowej')
plt.legend()
plt.grid(True)
plt.show()
