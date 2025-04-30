import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def generate_data(n_samples, n_features, random_state):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )
    y = 2 * y - 1 ## przekształcenie etykiet na {-1, 1}
    
    return X, y

def normalize_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled

def analytical_clf(X, y, alpha=0.001):
    y = np.array(y)
    n_samples, n_features = X.shape

    X_bias = np.hstack([X, np.ones((n_samples, 1))])

    I = np.eye(n_features + 1)
    I[-1, -1] = 0  # Brak regularyzacji dla biasu

    # Rozwiązanie analityczne: w = (X^T X + alpha I)^(-1) X^T y
    A = X_bias.T @ X_bias + alpha * I
    b = X_bias.T @ y
    solution = np.linalg.solve(A, b) ## rozwiązuje układ równań Aw = b

    return solution

def predict(X, solution):
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Dodaj kolumnę jedynki
    predictions = X @ solution
    return np.sign(predictions).astype(int) ## zwracanie wartości etykiet {-1, 1}


def sklearn_ridge_clf(X, y, alpha=0.001):
    clf = RidgeClassifier(alpha=alpha)

    clf.fit(X, y)
    predictions = clf.predict(X)

    return predictions


X, y = generate_data(100, 1, 195)
w = analytical_clf(X, y)
print(X, w)
predictions = predict(X, w)

print(accuracy_score(y, predictions))

sklearn_predictions = sklearn_ridge_clf(X, y)
print(accuracy_score(y, sklearn_predictions))