# Polynomial Regression From Scratch / Implantation From Scratch de la Regression Polynomiale

This repository contains a pure Python implementation (using only NumPy) of Polynomial Regression, featuring a detailed analysis of the critical impact of data normalization on gradient descent convergence.

Ce depot contient une implementation en Python pur (utilisant uniquement NumPy) de la regression polynomiale, mettant en evidence l'impact critique de la normalisation des données sur la convergence de la descente de gradient.

---

## English Version

### Project Features
* Flexible Model: A `PolynomialRegression` Python class mimicking the Scikit-Learn API (`fit`, `predict`, `score`).
* Dual Optimization Approach:
  * Matrix-based Gradient Descent.
  * Exact solution using the Normal Equation (Analytical Method).
* Performance Analysis: Real-time tracking of CPU time, iteration count, and R2 score.
* Automatic Visualization: Generation of comparative plots for polynomial fits and cost convergence.

### Mathematical Foundations

#### 1. Cost Function (Mean Squared Error)
For m samples and a Vandermonde matrix X, the cost function in matrix form is:
$$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$

#### 2. Gradient Derivation
$$\nabla J(\theta) = \frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (X\theta - y)$$

#### 3. Update Rule
$$\theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)$$

### Project Structure
The core model implementation is located in the `src/` directory, while the `experiments.py` script at the root runs simulations and saves performance plots into the `Figures/` folder.

### Key Insights: The Importance of Normalization
Without normalization, higher-degree polynomial features (e.g., d >= 5) cause the Vandermonde matrix columns to grow exponentially. This distorts the cost surface.
1. Without Normalization: The gradient descent requires an extremely small learning rate (\alpha = 0.001) to avoid divergence, leading to slow convergence.
2. With Normalization: A robust learning rate (\alpha = 0.05) can be applied, achieving global minimum convergence in fewer than 100 iterations.

---

## Version Francaise

### Fonctionnalites du Projet
* Modele Flexible : Classe Python `PolynomialRegression` mimant l'API Scikit-Learn (`fit`, `predict`, `score`).
* Double Approche d'Optimisation :
  * Descente de Gradient Matricielle (Gradient Descent).
  * Resolution Exacte via l'Equation Normale (Methode Analytique).
* Analyse de Performance : Suivi en temps reel du temps CPU, du nombre d'iterations et du score R2.
* Visualisation Automatique : Generation de graphiques comparatifs (Ajustements et courbes de convergence).

### Fondations Mathematiques

#### 1. Fonction Cout (Erreur Quadratique Moyenne)
Pour un ensemble de m echantillons et une matrice de Vandermonde X, la fonction de cout s'exprime par :
$$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$

#### 2. Derivation du Gradient
$$\nabla J(\theta) = \frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (X\theta - y)$$

#### 3. Regle de Mise a Jour
A chaque ituration, les parametres sont actualises selon le taux d'apprentissage (Learning Rate) \alpha :
$$\theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)$$

### Organisation du Projet
Le code source du modele se trouve dans le dossier `src/`, tandis que le script `experiments.py` a la racine permet de lancer les simulations et de generer les graphiques de performance dans le dossier `Figures/`.

### Analyses et Resultats Cles : L'Impact de la Normalisation
Sans normalisation, des que le degre du polynome augmente (d >= 5), les colonnes de la matrice de Vandermonde prennent des proportions gigantesques, ce qui etire la surface de cout.
1. Sans Normalisation : Le gradient requiert un taux d'apprentissage tres faible (\alpha = 0.001) pour eviter d'exploser, ce qui ralentit fortement la convergence.
2. Avec Normalisation : On peut utiliser un taux d'apprentissage robuste (\alpha = 0.05). La convergence vers le minimum global est atteinte en moins de 100 iterations.

---

## Installation & Usage / Installation et Utilisation

1. Clone the repository / Cloner le projet :
   ```bash
   git clone [https://github.com/toto-100/regression-polynomiale-from-scratch.git](https://github.com/toto-100/regression-polynomiale-from-scratch.git)
   cd regression-polynomiale-from-scratch