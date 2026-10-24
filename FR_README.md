# Modélisation mathématique d'un environnement dynamique

## Informations générales

- **Auteur :** Karl Garrido Andersson
- **Date :** 6 septembre 2026
- **Sujet :** Une approche fonctionnelle pour la quantification de l'« état de santé » d'un environnement dynamique

## Résumé

L'article propose un cadre mathématique formel pour modéliser un environnement dynamique et son évolution temporelle. Il construit une hiérarchie de trois objets fondamentaux : le **cadre**, les **événements** et l'**axe temporel**.

Chaque événement est décrit par cinq attributs quantifiables. Un opérateur **Γ** calcule sa gravité, en intégrant directement la durée restante dans la formule. Une **matrice d'interaction W(t)** capture l'amplification, la neutralisation et la rétroaction entre événements coexistants. Enfin, un opérateur **ζ** synthétise l'ensemble en un scalaire normalisé à valeurs dans (−1, 1), fondé sur `tanh` : l'état de santé de l'environnement.

Une expérience préliminaire sur données synthétiques teste les améliorations (dilatation, splines adaptatives, clusterisation) et identifie les limites actuelles du modèle.

## Problématique centrale

### Quantifier la santé d'un environnement

- **Objectif :** construire un indicateur normalisé de la propension d'un environnement à évoluer positivement ou négativement à un instant donné.
- **Difficulté :** les environnements sont divers, certains tangibles (une ville, une forêt), d'autres intangibles (un réseau social, un marché boursier).
- **Approche :** une représentation *événementielle*. L'environnement est entièrement décrit par ses événements constitutifs, sans modéliser explicitement le cadre spatial.
- **Applications visées :**
  - Finance : quantifier l'état d'un marché à partir d'événements macroéconomiques.
  - Écologie : évaluer la résilience d'un écosystème et estimer son point de bascule.
  - Gestion de risques : synthétiser des incidents opérationnels en un indicateur unique et actionnable.

### Les cinq attributs d'un événement

| Symbole | Nom | Description |
|---|---|---|
| D | Durée | Temps pendant lequel l'événement reste actif |
| F | Fréquence | Nombre d'occurrences par unité de temps |
| R | Rayon d'action | Étendue spatiale ou thématique de l'influence |
| G | Gravité | Intensité de l'impact intrinsèque |
| P | Prédictibilité | Degré auquel l'événement peut être anticipé |

## Solution proposée : le modèle fonctionnel

Le modèle est une chaîne de transformations :

```
événements → Γ (gravité) → (I + W(t)) (interactions) → somme pondérée (ω̄) → tanh(·/λ) → ζ ∈ (−1, 1)
```

1. **Gravité résiduelle temporelle (Γ).**
   `G(t) = max(D − (t − t₀), 0)^α · F^β · R^δ / (1 + P^γ)`.
   La gravité croît avec la durée, la fréquence et le rayon, décroît avec la prédictibilité, et s'annule exactement lorsque l'événement est terminé.
2. **Calibration par cluster.** Les événements sont regroupés en familles (financier, climatique, sanitaire...) et les exposants (α, β, δ, γ) sont ajustés par cluster en minimisant la MAPE.
3. **Clustering lexical.** Les événements sont regroupés selon leurs mots-clés communs, via la similarité de Jaccard avec un seuil τ.
4. **Matrice d'interaction W(t).** La gravité effective vaut `Ge(t) = (I + W(t)) G(t)`. Un coefficient positif traduit une amplification ou une rétroaction, un coefficient négatif une neutralisation.
5. **État de santé ζ.**
   `ζ = tanh( Σ ω̄ᵢ · Geᵢ / λ )`, avec des poids statiques normalisés en L¹.
   `tanh` borne la sortie, préserve le signe et encode l'impossibilité des extrêmes absolus.
6. **Dilatation λ.** Un paramètre qui contrôle l'échelle de saturation. La forme adaptative est `λ(t) = α N(t) / (1 + β |ζ(t−1)|)`, où N(t) est le nombre d'événements actifs.
7. **Drap variationnel de poids.** Une surface minimisant une énergie de Dirichlet normalisée, ajustée pour prédire le poids initial d'un nouvel événement.
8. **Lissage temporel.** Splines cubiques avec nœuds densifiés autour des discontinuités détectées.
9. **Réglage automatique (exploratoire).** Validation croisée K-fold et agent Q-Learning pour les hyperparamètres.
10. **Événements inconnus.** Un nouvel événement est représenté comme une combinaison lexicale pondérée de clusters existants (par exemple tsunami ≈ 0,6 séisme + 0,4 inondation).

## Détail des extensions

| Problème observé | Approche initiale | Correction introduite |
|---|---|---|
| ζ stagne à ±1 | λ constant = 1 | Dilatation, puis λ(t) adaptatif et pénalisé |
| Le lissage sort de [−1, 1] | Interpolation de Lagrange (phénomène de Runge) | Splines cubiques |
| Les fluctuations courtes sont manquées | Nœuds de spline uniformes | Nœuds adaptatifs aux discontinuités |
| Une seule loi pour tous les événements | Paramètres globaux | Calibration par cluster |
| Réglage manuel des hyperparamètres | Minimisation directe de la MAPE | Q-Learning (exploratoire) |

## Résultats préliminaires

Toutes les expériences utilisent une **base générée aléatoirement** (150 événements, 3 clusters : financier, climatique, sanitaire).

| Modèle | MAPE | R² |
|---|---|---|
| Global (sans cluster) | 4,32 % | 0,9913 |
| Par cluster | 4,10 % | n/a |
| Meilleur cluster (C3) | 3,35 % | n/a |

- Les exposants estimés ne diffèrent que modérément entre clusters (environ 4 % à 9 % d'écart relatif), la spécialisation portant surtout sur β et δ.
- La dilatation supprime l'essentiel de la saturation à ±1. Les valeurs λ ∈ {7,5 ; 10 ; 12,5} sont présentées comme un bon compromis.
- Les splines cubiques respectent le domaine (−1, 1), mais la version adaptative produit des creux parasites (notamment autour de t = 31 et t = 44), avec un écart maximal d'environ 1,0.

## Limites actuelles (selon l'auteur)

- **Subjectivité :** le cadre C, l'ordre de grandeur h et les poids initiaux ωᵢ sont des choix exogènes.
- **Clustering lexical :** le seuil τ est empirique. Trop bas, il fusionne des familles distinctes ; trop élevé, il fragmente des clusters homogènes.
- **Complexité :** le calcul de la gravité effective est en O(N²) dans l'implémentation naïve.
- **Drap variationnel :** sur ces données, il ne fait pas mieux qu'un modèle constant. Les poids ont été générés uniformément au hasard, indépendamment de (D, F, R, P), il n'y avait donc aucune structure à retrouver. Le critère pénalisé choisit correctement des degrés bas.
- **Q-Learning :** la discrétisation grossière de la MAPE (« point aveugle » et effet de bord) empêche l'agent d'atteindre la zone 0–4 %. Un approximateur continu de Q est suggéré.
- **Perte asymétrique** (sous-estimation pénalisée davantage) : définie mais pas encore intégrée à la calibration.
- **Aucune validation sur données réelles.**

## Perspectives

- **Court terme :** automatiser le choix de τ et de h, et intégrer la perte asymétrique dans la calibration par cluster.
- **Moyen terme :** remplacer la Q-table par un approximateur continu (Deep Q-Learning).
- **Long terme :** valider sur des jeux de données réels et automatiser l'opérateur Ψ (texte → attributs) avec un modèle de traitement du langage naturel.

## Origine des contributions

- **Originales (selon l'auteur) :** la forme de Γ avec gravité résiduelle temporelle, la calibration par cluster, ζ avec dilatation, le drap variationnel de poids, les nœuds adaptatifs.
- **Issues de la littérature :** propriétés de `tanh`, k-means, Q-Learning, splines cubiques, fonctionnelle de Dirichlet.
