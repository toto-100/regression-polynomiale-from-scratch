import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000, method='gradient'):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.method = method
        self.theta = None
        self.cost_history = []

    def polynomial_features(self, X):
        """Transforme un vecteur X de taille (m,) en matrice de Vandermonde de taille (m, degree + 1)"""
        X_flat = X.flatten()
        # Vectorisation pure : élévation à la puissance par broadcasting
        powers = np.arange(self.degree + 1)
        X_poly = X_flat[:, np.newaxis] ** powers
        return X_poly

    def fit(self, X, y):
        X_poly = self.polynomial_features(X)
        m, n = X_poly.shape
        
        # Sécurité pour s'assurer que y est un vecteur colonne propre
        y_clean = y.reshape(-1, 1)
        
        if self.method == 'analytic':
            # Équation Normale : (X^T * X)^(-1) * X^T * y
            self.theta = np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y_clean)
            errors = X_poly.dot(self.theta) - y_clean
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history = [cost]
        
        elif self.method == 'gradient':
            self.theta = np.zeros((n, 1))
            self.cost_history = []
            
            for iteration in range(self.iterations):
                predictions = X_poly.dot(self.theta)
                errors = predictions - y_clean
                gradient = (1 / m) * X_poly.T.dot(errors)
                self.theta -= self.learning_rate * gradient

                cost = (1 / (2 * m)) * np.sum(errors ** 2)
                self.cost_history.append(cost)

                if np.isnan(cost) or np.isinf(cost):
                    print(f"⚠️ [Degré {self.degree}] Le gradient a explosé à l'itération {iteration}. Normalisation ou LR plus faible requis !")
                    break
        else:
            raise ValueError("Méthode inconnue. Choisissez 'gradient' ou 'analytic'.")

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        return X_poly.dot(self.theta).flatten()

    def score(self, X, y):
        """Calcule le coefficient de détermination R²"""
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - (u / v)