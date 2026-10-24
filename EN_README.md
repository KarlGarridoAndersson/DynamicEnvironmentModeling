# Mathematical Modeling of a Dynamic Environment

## General Information

- **Author:** Karl Garrido Andersson
- **Date:** September 6, 2026
- **Subject:** A functional approach to quantifying the "health state" of a dynamic environment

## Abstract

The article proposes a formal mathematical framework for modeling a dynamic environment and its evolution over time. It builds a hierarchy of three fundamental objects: the **frame**, the **events** and the **time axis**.

Each event is described by five quantifiable attributes. An operator **Γ** computes its gravity, with the remaining duration built directly into the formula. An **interaction matrix W(t)** captures amplification, neutralization and feedback between coexisting events. Finally, an operator **ζ** synthesizes everything into a single normalized scalar in (−1, 1), based on `tanh`: the environment's health state.

A preliminary experiment on synthetic data tests the improvements (dilation, adaptive splines, clustering) and identifies the model's current limits.

## Central Problem

### Quantifying the health of an environment

- **Goal:** build a normalized indicator of an environment's propensity to evolve positively or negatively at a given time.
- **Difficulty:** environments are diverse, some tangible (a city, a forest) and some intangible (a social network, a stock market).
- **Approach:** an *event-based* representation. The environment is described entirely by its constituent events, without explicitly modeling the spatial frame.
- **Target applications:**
  - Finance: quantify a market's state from macroeconomic events.
  - Ecology: assess an ecosystem's resilience and estimate its tipping point.
  - Risk management: summarize operational incidents in a single actionable indicator.

### The five attributes of an event

| Symbol | Name | Description |
|---|---|---|
| D | Duration | Time during which the event stays active |
| F | Frequency | Occurrences per unit of time |
| R | Radius | Spatial or thematic extent of influence |
| G | Gravity | Intrinsic impact intensity |
| P | Predictability | Degree to which the event can be anticipated |

## Proposed Solution: The Functional Model

The model is a chain of transformations:

```
events → Γ (gravity) → (I + W(t)) (interactions) → weighted sum (ω̄) → tanh(·/λ) → ζ ∈ (−1, 1)
```

1. **Residual temporal gravity (Γ).**
   `G(t) = max(D − (t − t₀), 0)^α · F^β · R^δ / (1 + P^γ)`.
   Gravity grows with duration, frequency and radius, decreases with predictability, and vanishes exactly once the event has expired.
2. **Per-cluster calibration.** Events are grouped into families (financial, climatic, health...) and the exponents (α, β, δ, γ) are fitted per cluster by minimizing the MAPE.
3. **Lexical clustering.** Events are grouped by shared keywords using Jaccard similarity with a threshold τ.
4. **Interaction matrix W(t).** Effective gravity is `Ge(t) = (I + W(t)) G(t)`. A positive entry means amplification or feedback, a negative one means neutralization.
5. **Health state ζ.**
   `ζ = tanh( Σ ω̄ᵢ · Geᵢ / λ )`, with static weights normalized in L¹.
   `tanh` bounds the output, preserves the sign, and encodes the impossibility of absolute extremes.
6. **Dilation λ.** A parameter that controls the saturation scale. The adaptive form is `λ(t) = α N(t) / (1 + β |ζ(t−1)|)`, where N(t) is the number of active events.
7. **Variational weight sheet.** A surface minimizing a normalized Dirichlet energy, fitted to predict the initial weight of a new event.
8. **Temporal smoothing.** Cubic splines with knots densified around detected discontinuities.
9. **Automatic tuning (exploratory).** K-fold cross-validation and a Q-Learning agent for the hyperparameters.
10. **Unknown events.** A new event is represented as a weighted lexical combination of existing clusters (e.g. tsunami ≈ 0.6 earthquake + 0.4 flood).

## Breaking Down the Extensions

| Problem observed | Initial approach | Fix introduced |
|---|---|---|
| ζ stagnates at ±1 | Constant λ = 1 | Dilation, then adaptive and penalized λ(t) |
| Smoothing leaves [−1, 1] | Lagrange interpolation (Runge phenomenon) | Cubic splines |
| Short fluctuations are missed | Uniform spline knots | Adaptive knots at discontinuities |
| One law for all events | Global parameters | Per-cluster calibration |
| Manual hyperparameter tuning | Direct MAPE minimization | Q-Learning (exploratory) |

## Preliminary Results

All experiments use a **randomly generated synthetic base** (150 events, 3 clusters: financial, climatic, health).

| Model | MAPE | R² |
|---|---|---|
| Global (no cluster) | 4.32 % | 0.9913 |
| Per cluster | 4.10 % | n/a |
| Best cluster (C3) | 3.35 % | n/a |

- Fitted exponents differ only moderately between clusters (about 4 % to 9 % relative gap), with the differences mostly in β and δ.
- Dilation removes most of the saturation at ±1. The values λ ∈ {7.5, 10, 12.5} are described as a good compromise.
- Cubic splines respect the (−1, 1) domain, but the adaptive version produces spurious dips (notably around t = 31 and t = 44), with a maximum deviation of about 1.0.

## Current Limits (as stated by the author)

- **Subjectivity:** the frame C, the order of magnitude h and the initial weights ωᵢ are exogenous choices.
- **Lexical clustering:** the threshold τ is empirical. Too low merges distinct families, too high fragments homogeneous ones.
- **Complexity:** computing effective gravity is O(N²) in the naive implementation.
- **Variational sheet:** on this data it does no better than a constant model. The weights were generated uniformly at random, independently of (D, F, R, P), so there was no structure to recover. The penalized criterion correctly picks low degrees.
- **Q-Learning:** coarse discretization of the MAPE ("blind spot" and edge effect) prevents the agent from reaching the 0–4 % zone. A continuous Q approximator is suggested.
- **Asymmetric loss** (under-estimation penalized more heavily) is defined but not yet integrated into the calibration.
- **No validation on real data.**

## Perspectives

- **Short term:** automate the choice of τ and of h, and integrate the asymmetric loss into the per-cluster calibration.
- **Medium term:** replace the Q-table with a continuous approximator (Deep Q-Learning).
- **Long term:** validate on real datasets and automate the operator Ψ (text → attributes) with a natural language processing model.

## Origin of the Contributions

- **Original (per the author):** the form of Γ with residual temporal gravity, per-cluster calibration, ζ with dilation, the variational weight sheet, adaptive knots.
- **From the literature:** properties of `tanh`, k-means, Q-Learning, cubic splines, the Dirichlet functional.
