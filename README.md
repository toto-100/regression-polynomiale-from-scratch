# Implantation From Scratch de la RÃ©gression Polynomiale ðŸ“ŠðŸ“ˆ

Ce projet propose une implÃ©mentation complÃ¨te en **Python pur (utilisant uniquement NumPy)** de la rÃ©gression polynomiale. DÃ©veloppÃ© dans le cadre d'une Ã©tude thÃ©orique et pratique, il dÃ©taille les dÃ©rivations mathÃ©matiques sous-jacentes (fonction coÃ»t, calcul du gradient) et analyse l'impact critique de la normalisation sur la convergence et l'utilisation des ressources CPU.

## ðŸš€ FonctionnalitÃ©s du Projet
* **ModÃ¨le Flexible :** Classe Python `PolynomialRegression` mimant l'API Scikit-Learn (`fit`, `predict`, `score`).
* **Double Approche d'Optimisation :**
  * Descente de Gradient Matricielle (Gradient Descent).
  * RÃ©solution Exacte via l'Ã‰quation Normale (MÃ©thode Analytique).
* **Analyse de Performance :** Suivi en temps rÃ©el du temps CPU, du nombre d'itÃ©rations et du score $R^2$.
* **Visualisation Automatique :** GÃ©nÃ©ration automatique de graphiques comparatifs (Ajustements et courbes de convergence).

---

## ðŸ“ Fondations MathÃ©matiques

### 1. Fonction CoÃ»t (Erreur Quadratique Moyenne)
Pour un ensemble de $m$ Ã©chantillons et une matrice de Vandermonde $X$ (contenant les puissances de nos variables), la fonction de coÃ»t sous forme matricielle s'exprime par :

$$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$

### 2. DÃ©rivation du Gradient
Pour ajuster nos paramÃ¨tres $\theta$ via la descente de gradient, nous calculons la dÃ©rivÃ©e partielle de $J(\theta)$ par rapport Ã  $\theta$. La dÃ©rivation complÃ¨te donne :

$$\nabla J(\theta) = \frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T (X\theta - y)$$

### 3. RÃ¨gle de Mise Ã  Jour
Ã€ chaque itÃ©ration, les paramÃ¨tres sont actualisÃ©s selon le taux d'apprentissage (Learning Rate) $\alpha$ :

$$\theta \leftarrow \theta - \alpha \cdot \nabla J(\theta)$$

---

## ðŸ“‚ Organisation du Projet
Le code source du modÃ¨le se trouve dans le dossier `src/`, tandis que le script `experiments.py` Ã  la racine permet de lancer les simulations et de gÃ©nÃ©rer les graphiques de performance dans le dossier `Figures/`.

---

## ðŸ“Š Analyses & RÃ©sultats ClÃ©s

### ðŸ’¡ L'Impact Crucial de la Normalisation
Sans normalisation (Z-score ou MinMax), dÃ¨s que le degrÃ© du polynÃ´me augmente (par exemple $d \ge 5$), les colonnes de la matrice de Vandermonde prennent des proportions gigantesques ($X^5$). Cela Ã©tire la surface de coÃ»t de maniÃ¨re disproportionnÃ©e. 

Le graphique de convergence met en Ã©vidence deux comportements :
1. **Sans Normalisation :** Le gradient requiert un taux d'apprentissage extrÃªmement faible ($\alpha = 0.001$) pour Ã©viter d'exploser, ce qui ralentit fortement la convergence.
2. **Avec Normalisation :** On peut utiliser un taux d'apprentissage robuste ($\alpha = 0.05$). La convergence vers le minimum global est atteinte en moins de 100 itÃ©rations.

*Les graphiques comparatifs d'ajustements et de trajectoire de coÃ»t sont sauvegardÃ©s automatiquement dans le dossier `Figures/` aprÃ¨s exÃ©cution.*

---

## ðŸ› ï¸ Installation et Utilisation

1. **Cloner le projet :**
   ```bash
   git clone [https://github.com/TON_PSEUDO/Regression_Polynomiale_From_Scratch.git](https://github.com/TON_PSEUDO/Regression_Polynomiale_From_Scratch.git)
   cd Regression_Polynomiale_From_Scratch
