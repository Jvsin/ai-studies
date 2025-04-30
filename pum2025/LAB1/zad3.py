import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pandas as pd

## full data
full_data = pd.read_csv('full_path.csv', header=None)
X_full = full_data[0].values.reshape(-1, 1)
y_full = full_data[1].values
print(len(X_full), len(y_full))

## min road
min_data = pd.read_csv('min.csv', header=None)
X_min = min_data[0].values.reshape(-1, 1)
y_min = min_data[1].values
print(len(X_min), len(y_min))

## key moments
X_key, y_key = [], []
last_y = None
for i, (x, y) in enumerate(zip(X_full.flatten(), y_full)):
    if y != last_y or (i > 0 and y_full[i-1] != y):
        X_key.append(x)
        y_key.append(y)
        last_y = y
X_key = np.array(X_key).reshape(-1, 1)
y_key = np.array(y_key)
print(len(X_key), len(y_key))

def mse(w, X, y):
    X_bias = np.vstack([X, np.ones(len(X))]).T
    results = X_bias @ w
    
    return np.sum((y - results) ** 2) / len(y)

def normalize_data_min_max(X, y):
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled

def normalize_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # scaler_y = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled

def analytical_ridge_regression(X, y, degree, alpha=1.0):
    X_scaled, y_scaled = normalize_data_min_max(X, y)

    X_poly = np.ones((len(X_scaled), degree + 1))
    for d in range(degree + 1):
        X_poly[:, d] = X_scaled.flatten() ** d
     
    I = np.eye(degree + 1)
    w = np.linalg.inv(X_poly.T @ X_poly + alpha * I) @ X_poly.T @ y_scaled

    predictions = X_poly @ w

    mse = mean_squared_error(y_scaled, predictions)
    print(mse)
    plt.scatter(X_scaled, y_scaled, color='blue', label="Dane rzeczywiste")
    plt.plot(X_scaled, predictions, color='red', label=f"Regresja (stopień {degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Regresja analityczna")
    plt.grid(True)
    plt.show()

def sklearn_ridge_regression(X, y, degree, alpha=1.0):
    X_scaled, y_scaled= normalize_data_min_max(X, y)
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)
    
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y_scaled)
    
    predictions = model.predict(X_poly)
    
    mse = mean_squared_error(y_scaled, predictions)
    print(mse)
    plt.scatter(X_scaled, y_scaled, color='blue', label="Dane rzeczywiste")
    plt.plot(X_scaled, predictions, color='red', label=f"Regresja (stopień {degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Regresja sklearn")
    plt.grid(True)
    plt.show()

analytical_ridge_regression(X_full, y_full, 21)
sklearn_ridge_regression(X_full, y_full, 21)

analytical_ridge_regression(X_key, y_key, 21)
sklearn_ridge_regression(X_key, y_key, 21)

analytical_ridge_regression(X_min, y_min, 21)
sklearn_ridge_regression(X_min, y_min, 21)
    