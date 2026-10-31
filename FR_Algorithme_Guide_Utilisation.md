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
