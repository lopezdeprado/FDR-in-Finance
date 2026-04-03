# FDR in Finance

Research code implementing **False Discovery Rate (FDR)** methods for financial asset
return predictors, addressing the multiple-testing problem that arises when researchers
evaluate hundreds of candidate equity predictors on the same dataset.

## Background

When testing many predictors simultaneously, standard significance thresholds (e.g.
*p* < 0.05) become unreliable because the selected test statistic is the maximum over
*K* correlated draws — not a single unbiased draw. This repository provides:

- A **maximum-of-mixtures likelihood framework** to model the distribution of selected
  test statistics.
- Code to reproduce **Table 1** (Section 4.4.3) and **Table 2** (Section 6) from the
  accompanying paper.

## Repository Structure

```
FDR-in-Finance/
├── PredictorLSretWide.csv               # Monthly returns for ~100 equity predictors
├── code_table_1.py                      # Reproduces Table 1 (conditional tail probs)
└── code_table_2-robust-parallel.py      # Reproduces Table 2 (FDR from real data)
```

## Usage

### Table 1 — Conditional Tail Probabilities (Section 4.4.3)

Compares the conditional tail probability *P(X ≥ x | X ≥ c)* under two scenarios:

| Scenario | Description |
|----------|-------------|
| **Case A** (with search) | π₀ = 0.75, K = 5, SR₁ = 0.30; selected statistic is max over K draws |
| **Case B** (no search)   | π₀ = 0.10, K = 1, SR₁ = 0.30; single hypothesis test |

```bash
python code_table_1.py
```

Outputs:
- `table1_conditional_tail_probabilities.csv`
- `table1_conditional_tail_probabilities_unrounded.csv`

### Table 2 — FDR Estimation from Real Data (Section 6)

Loads the predictor dataset, estimates Sharpe ratios and AR(1) autocorrelations,
then fits a maximum-of-mixtures model for each value of *K* (1–10, 25, 50, 75, 100).
Parallelized across *K* values via `ProcessPoolExecutor`.

```bash
python code_table_2-robust-parallel.py
```

Outputs:
- `Table2_sigma1_ge_sigma0_monthly_strongopt_parallel.csv` — main results table
- `Section6_predictor_stats_monthly.csv` — per-predictor statistics

## Statistical Framework

The distribution of the *selected* test statistic (the maximum of *K* draws) is
modelled as a **maximum-of-mixtures**:

```
F(x) = [ π₀ · Φ(x/σ₀)  +  (1-π₀) · Φ((x-δ₁)/σ₁) ]^K
```

where:
- **π₀** — proportion of true null hypotheses
- **δ₁** — mean shift under the alternative (effect size)
- **σ₀, σ₁** — standard deviations under the null and alternative (σ₁ ≥ σ₀)

Parameters are estimated by maximum likelihood. The fitted model yields per-predictor
familywise error rates (α_K, β_K) and the **False Discovery Rate**:

```
FDR = (ᾱ_K · π₀) / (ᾱ_K · π₀  +  (1 − β̄_K) · (1 − π₀))
```

## Dependencies

- Python 3.x
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
