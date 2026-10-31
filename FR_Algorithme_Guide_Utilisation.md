# Guide d'Utilisation : Modélisation de l'État de Santé d'un Environnement Dynamique

Ce guide détaille le fonctionnement technique, la manipulation des variables et l'exécution de l'apprentissage itératif pour la modélisation mathématique d'un environnement dynamique.

---

## 1. Configuration de l'Instance

Le pilotage de la simulation et de l'apprentissage s'effectue via des constantes globales définies au début du notebook. Modifier ces valeurs permet d'ajuster la taille du jeu de données, la structure des événements et les hyperparamètres d'optimisation.

| Variable | Type | Description |
| :---: | :--- | :--- |
| `RANDOM_SEED` | `int` | Graine aléatoire pour figer la reproductibilité des résultats. |
| `N_EVENTS` | `int` | Nombre total d'événements générés dans la base de données. |
| `SIMILARITY_RATIO` | `float` | Proportion d'événements générés via des clusters contrôlés. |
| `TRUE_PARAMS` | `tuple` | Paramètres cibles $(\alpha, \beta, \delta, \gamma)$ utilisés pour générer les données synthétiques. |
| `N_CLUSTERS` | `int` | Nombre de groupes pour le clustering K-Means. |
| `N_ITERATIONS` | `int` | Nombre d'itérations pour la boucle d'apprentissage itérative globale. |
| `N_SPLITS` | `int` | Nombre de plis (*folds*) pour la validation croisée `KFold`. |
| `UPDATE_RATE` | `float` | Taux de mise à jour (*learning rate*) des paramètres d'une itération à l'autre. |

---

## 2. Manipulation des Fonctions & Sections

### 2.1 Fonctions du Modèle Mathématique
Les fonctions clés permettent de calculer la gravité d'un événement, la métrique d'erreur (MAPE) et l'état de santé global $\zeta$.

```python
# Calcul de la gravité intrinsèque d'un événement
def gravity(D, F, R, P, alpha, beta, delta, gamma):
    """
    G = D^alpha * F^beta * R^delta / (1 + P^gamma)
    """
    num = np.abs(D)**alpha * np.abs(F)**beta * np.abs(R)**delta
    den = 1.0 + np.abs(P)**gamma
    return num / den

# Calcul de l'opérateur d'état de santé avec poids normalisés L1
def health_operator(G_eff, weights):
    """
    ζ = tanh(⟨ω, G_eff⟩), avec normalisation L1 des poids ω.
    """
    w = np.array(weights, dtype=float)
    norm = np.sum(np.abs(w))
    if norm < 1e-12:
        return 0.0
    w = w / norm
    return float(np.tanh(np.dot(w, G_eff)))
```

### 2.2 Point d'Entrée Principal d'Apprentissage
La fonction `run_iterative_learning()` orchestre l'optimisation des paramètres $\Gamma = (\alpha, \beta, \delta, \gamma)$ en combinant une optimisation `L-BFGS-B` et une validation croisée à $K$ plis.

```python
def run_iterative_learning(D_tr, F_tr, R_tr, P_tr, G_tr,
                           D_te, F_te, R_te, P_te, G_te,
                           n_iter=10, n_splits=5, update_rate=0.5,
                           verbose=True):
    """
    Exécute l'apprentissage itératif avec validation croisée (K-Fold).
    Retourne l'historique des paramètres et des erreurs MAPE (train/test).
    """
```

---

## 3. Structure des Sections du Notebook

| Section | Titre | Description |
| :---: | :--- | :--- |
| **1** | Imports & Configuration | Chargement des bibliothèques (`numpy`, `scipy`, `sklearn`), fixation du seed et définition des styles graphiques. |
| **2** | Modèle Mathématique | Définition des équations de gravité $G$, résiduelle $G(t)$, opérateur de santé $\zeta$ et métrique MAPE. |
| **3** | Base de Données & Clustering | Génération des 150 événements synthétiques et clustering K-Means ($K=3$) sur $(D, F, R, P)$. |
| **4** | Apprentissage Itératif Global | Optimisation des paramètres globaux $(\alpha, \beta, \delta, \gamma)$ via validation croisée (5-Fold). |
| **5–8**| Figures & Diagnostics | Génération et sauvegarde des graphiques d'évaluation (convergence des paramètres, distributions MAPE, résidus). |
| **9–10**| Apprentissage par Clusters | Entraînement de jeux de paramètres spécifiques à chaque cluster d'événements. |
| **11–17**| Extensions & Modélisation Avancée | Poids statiques $L_1$, dilatation $\lambda$, lissage par Spline, Q-Learning et surfaces Plotly 3D interactives. |
| **A–E**| Annexes Théoriques | Implémentations d'extinction temporelle, drap variationnel, et décomposition hybride d'événements. |
