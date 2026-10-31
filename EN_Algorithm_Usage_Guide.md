# User Guide: Dynamic Environment Health State Modeling

This guide details the technical operation, variable manipulation, and execution of iterative learning for the mathematical modeling of a dynamic environment.

---

## 1. Instance Configuration

Simulation and learning parameters are controlled via global constants defined at the beginning of the notebook. Modifying these values adjusts dataset size, event structure, and optimization hyperparameters.

| Variable | Type | Description |
| :---: | :--- | :--- |
| `RANDOM_SEED` | `int` | Random seed for reproducibility of results. |
| `N_EVENTS` | `int` | Total number of events generated in the database. |
| `SIMILARITY_RATIO` | `float` | Proportion of events generated via controlled clusters. |
| `TRUE_PARAMS` | `tuple` | Target parameters $(\alpha, \beta, \delta, \gamma)$ used to generate synthetic data. |
| `N_CLUSTERS` | `int` | Number of groups for K-Means clustering. |
| `N_ITERATIONS` | `int` | Number of iterations for the global iterative learning loop. |
| `N_SPLITS` | `int` | Number of folds for `KFold` cross-validation. |
| `UPDATE_RATE` | `float` | Parameter update rate (learning rate) between iterations. |

---

## 2. Functions & Section Handling

### 2.1 Mathematical Model Functions
Key functions calculate event gravity, Error Metric (MAPE), and overall health state $\zeta$.

```python
# Calculation of intrinsic gravity of an event
def gravity(D, F, R, P, alpha, beta, delta, gamma):
    """
    G = D^alpha * F^beta * R^delta / (1 + P^gamma)
    """
    num = np.abs(D)**alpha * np.abs(F)**beta * np.abs(R)**delta
    den = 1.0 + np.abs(P)**gamma
    return num / den

# Calculation of the health operator with L1-normalized weights
def health_operator(G_eff, weights):
    """
    ζ = tanh(⟨ω, G_eff⟩), with L1 normalization of weights ω.
    """
    w = np.array(weights, dtype=float)
    norm = np.sum(np.abs(w))
    if norm < 1e-12:
        return 0.0
    w = w / norm
    return float(np.tanh(np.dot(w, G_eff)))
```

### 2.2 Main Learning Entry Point
The `run_iterative_learning()` function orchestrates the optimization of parameters $\Gamma = (\alpha, \beta, \delta, \gamma)$ by combining `L-BFGS-B` optimization with $K$-fold cross-validation.

```python
def run_iterative_learning(D_tr, F_tr, R_tr, P_tr, G_tr,
                           D_te, F_te, R_te, P_te, G_te,
                           n_iter=10, n_splits=5, update_rate=0.5,
                           verbose=True):
    """
    Runs iterative learning with K-Fold cross-validation.
    Returns parameter history and MAPE error metrics (train/test).
    """
```

---

## 3. Notebook Section Structure

| Section | Title | Description |
| :---: | :--- | :--- |
| **1** | Imports & Configuration | Library imports (`numpy`, `scipy`, `sklearn`), setting seeds, and defining plot styles. |
| **2** | Mathematical Model | Definition of gravity equations $G$, residual $G(t)$, health operator $\zeta$, and MAPE metric. |
| **3** | Database & Clustering | Synthetic generation of 150 events and K-Means clustering ($K=3$) on $(D, F, R, P)$. |
| **4** | Global Iterative Learning | Optimization of global parameters $(\alpha, \beta, \delta, \gamma)$ via 5-Fold cross-validation. |
| **5–8**| Figures & Diagnostics | Generation and saving of diagnostic plots (parameter convergence, MAPE distributions, residuals). |
| **9–10**| Cluster-based Learning | Training specific parameter sets per event cluster. |
| **11–17**| Extensions & Advanced Modeling | Static $L_1$ weights, dilation $\lambda$, Spline smoothing, Q-Learning, and interactive 3D Plotly surfaces. |
| **A–E**| Theoretical Annexes | Implementations of temporal decay, variational drape, and hybrid event decomposition. |
