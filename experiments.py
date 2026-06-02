import matplotlib.pyplot as plt
import time
import os
import numpy as np
from polynomial_regression import PolynomialRegression
from utils import generate_data, standardize_data

# Créer le dossier pour les figures s'il n'existe pas
os.makedirs("Figures", exist_ok=True)

# 1. Préparation des données
X_raw, y = generate_data()
X_norm = standardize_data(X_raw)

degrees = [1, 2, 3, 5]

print("\n=== ANALYSE DES PERFORMANCES (MÉTHODE GRADIENT) ===")
print(f"{'Degré':<8}{'Normalisé':<12}{'Temps CPU (s)':<18}{'Score R²':<10}{'Coût Final':<12}")
print("-" * 60)

# =====================================================================
# 1. COMPARAISON DES COURBES D'AJUSTEMENT (CÔTE À CÔTE)
# =====================================================================
for degree in degrees:
    # Création d'une figure avec 2 graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- GAUCHE : Non-Normalisé ---
    model_raw = PolynomialRegression(degree=degree, learning_rate=0.001, iterations=4000)
    
    start_time = time.perf_counter()
    model_raw.fit(X_raw, y)
    cpu_time_raw = time.perf_counter() - start_time
    
    pred_raw = model_raw.predict(X_raw)
    r2_raw = model_raw.score(X_raw, y)
    cost_raw = model_raw.cost_history[-1] if model_raw.cost_history else np.nan
    
    print(f"{degree:<8}{'NON':<12}{cpu_time_raw:.6f}{'':<4}{r2_raw:.4f}{'':<4}{cost_raw:.2f}")
    
    sorted_idx_raw = X_raw.argsort()
    ax1.scatter(X_raw, y, color="gray", alpha=0.6, label="Données brutes")
    ax1.plot(X_raw[sorted_idx_raw], pred_raw[sorted_idx_raw], color="red", linewidth=2, label=f"Fit (R²={r2_raw:.2f})")
    ax1.set_title(f"Non-Normalisé (LR=0.001) - Degré {degree}")
    ax1.set_xlabel("X brut")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend()

    # --- DROITE : Normalisé ---
    model_norm = PolynomialRegression(degree=degree, learning_rate=0.05, iterations=4000)
    
    start_time = time.perf_counter()
    model_norm.fit(X_norm, y)
    cpu_time_norm = time.perf_counter() - start_time
    
    pred_norm = model_norm.predict(X_norm)
    r2_norm = model_norm.score(X_norm, y)
    cost_norm = model_norm.cost_history[-1] if model_norm.cost_history else np.nan
    
    print(f"{degree:<8}{'OUI':<12}{cpu_time_norm:.6f}{'':<4}{r2_norm:.4f}{'':<4}{cost_norm:.2f}")
    print("-" * 60)
    
    sorted_idx_norm = X_norm.argsort()
    ax2.scatter(X_norm, y, color="blue", alpha=0.4, label="Données normalisées")
    ax2.plot(X_norm[sorted_idx_norm], pred_norm[sorted_idx_norm], color="red", linewidth=2, label=f"Fit (R²={r2_norm:.2f})")
    ax2.set_title(f"Normalisé (LR=0.05) - Degré {degree}")
    ax2.set_xlabel("X standardisé (Z-score)")
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle(f"Impact de la Normalisation - Régression Polynomiale (Degré {degree})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 💾 Étape 1 : On sauvegarde l'image dans le dossier Figures
    plt.savefig(f"Figures/comparaison_ajustement_deg_{degree}.png")
    
    # 👁️ Étape 2 : On affiche le graphe à l'écran. 
    # (Le script fera une pause. Dès que tu fermes la fenêtre, il passe au degré suivant)
    plt.show()

# =====================================================================
# 2. COMPARAISON DE LA CONVERGENCE (CÔTE À CÔTE)
# =====================================================================
print("\nGénération du graphique de convergence globale...")
fig_conv, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 5))

for degree in [1, 2, 3, 5]:
    m_raw = PolynomialRegression(degree=degree, learning_rate=0.001, iterations=1500)
    m_raw.fit(X_raw, y)
    if m_raw.cost_history:
        ax_c1.plot(m_raw.cost_history, label=f"Degré {degree}")
        
    m_norm = PolynomialRegression(degree=degree, learning_rate=0.05, iterations=1500)
    m_norm.fit(X_norm, y)
    if m_norm.cost_history:
        ax_c2.plot(m_norm.cost_history, label=f"Degré {degree}")

ax_c1.set_title("Convergence SANS Normalisation (LR=0.001)")
ax_c1.set_ylabel("Coût (MSE)")
ax_c1.set_xlabel("Itérations")
ax_c1.grid(True)
ax_c1.legend()

ax_c2.set_title("Convergence AVEC Normalisation (LR=0.05)")
ax_c2.set_xlabel("Itérations")
ax_c2.grid(True)
ax_c2.legend()

plt.suptitle("Vitesse de convergence du Gradient Descent", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Figures/comparaison_convergence.png")
plt.show()