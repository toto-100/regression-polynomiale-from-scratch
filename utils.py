import numpy as np

def generate_data(n_samples=100):
    """Génère un jeu de données synthétique non linéaire."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    # Fonction cubique de base
    y = 0.5 * X**3 - 2 * X**2 + X + 3
    # Ajout d'un bruit gaussien
    noise = np.random.randn(n_samples) * 3
    y = y + noise
    return X, y

def standardize_data(X):
    """Normalisation standard (Z-score) : (X - mean) / std"""
    return (X - np.mean(X)) / np.std(X)