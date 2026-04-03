# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "pandas==2.2.3",
#   "scipy==1.15.3",
# ]
# ///

# ============================================================
# Replication code for Section 6 and Table 2
# Unequal-variance version with sigma_1 >= sigma_0
# NO ANNUALIZATION: everything is kept in monthly Sharpe units
#
# Parallelized version:
#   - parallelizes across K with multiprocessing
#   - keeps each K fit serial by default to avoid oversubscription
#   - optional switch to use SciPy differential_evolution workers
#
# Outputs:
#   - Table2_sigma1_ge_sigma0_monthly_strongopt_parallel.csv
#   - Section6_predictor_stats_monthly.csv
# ============================================================

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, minimize
from scipy.special import ndtr
from scipy.stats import norm

# ------------------------------------------------------------
# 1. USER SETTINGS
# ------------------------------------------------------------

CSV_PATH = "PredictorLSretWide.csv"
ALPHA = 0.05
K_GRID = list(range(1, 11))+[25,50,75,100]

# Optimizer settings
DE_MAXITER = 40
DE_POPSIZE = 12
DE_SEEDS = [101, 202, 303]
LOCAL_MAXITER = 5000

# Lower bounds on scale parameters.
# These are not part of the statistical model; they are numerical safeguards
# against near-singular Gaussian-mixture solutions.
SIGMA0_MIN = 0.02
SIGMA_EXTRA_MIN = 0.001

# Parallel settings
# Main recommendation: parallelize across K and keep DE serial within each K.
PARALLEL_OVER_K = True
MAX_WORKERS_K = min(len(list(K_GRID)), os.cpu_count() or 1)
DE_WORKERS = 1  # set to -1 only if PARALLEL_OVER_K = False

# ------------------------------------------------------------
# 2. HELPER FUNCTIONS: SAMPLE STATISTICS PER PREDICTOR
# ------------------------------------------------------------

def raw_sharpe(x: pd.Series) -> float:
    """
    Compute the raw (monthly, non-annualized) sample Sharpe ratio:
        SR = mean / std
    """
    x = x.dropna().astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    return mu / sd


def ar1_autocorr(x: pd.Series) -> float:
    """
    First-order sample autocorrelation.
    """
    x = x.dropna().astype(float)
    return x.autocorr(lag=1)


def ar1_threshold(T: int, rho: float, alpha: float = 0.05) -> float:
    """
    AR(1)-adjusted rejection threshold in raw monthly Sharpe-ratio units.

    Under H0: SR = 0,
        SR_hat ~ N(0, (1/T) * (1+rho)/(1-rho))

    Therefore the two-sided threshold is:
        c = z_(1-alpha/2) * sqrt((1+rho)/((1-rho) * T))
    """
    zcrit = norm.ppf(1 - alpha / 2)
    return zcrit * np.sqrt((1 + rho) / ((1 - rho) * T))


# ------------------------------------------------------------
# 3. BUILD THE CROSS-SECTION USED IN SECTION 6
# ------------------------------------------------------------

def build_cross_section(csv_path: str):
    df = pd.read_csv(csv_path)
    ret_df = df.drop(columns=["date"], errors="ignore")

    rows = []
    for name in ret_df.columns:
        s = ret_df[name].dropna().astype(float)

        T = len(s)
        if T < 3:
            continue

        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            continue

        sr = raw_sharpe(s)
        rho = ar1_autocorr(s)

        # Guard against pathological autocorrelation estimates near +/-1
        if not np.isfinite(rho):
            continue
        rho = np.clip(rho, -0.99, 0.99)

        c = ar1_threshold(T=T, rho=rho, alpha=ALPHA)

        rows.append(
            {
                "predictor": name,
                "T": T,
                "rho": rho,
                "SR_hat": sr,
                "c_n": c,
            }
        )

    stats_df = pd.DataFrame(rows)
    x_obs = stats_df["SR_hat"].to_numpy(dtype=float)
    c_vec = stats_df["c_n"].to_numpy(dtype=float)
    return stats_df, x_obs, c_vec


# ------------------------------------------------------------
# 4. MAXIMUM-OF-MIXTURES LIKELIHOOD
# ------------------------------------------------------------

def logistic(a: float) -> float:
    """Map R to (0,1)."""
    return 1.0 / (1.0 + np.exp(-a))


def unpack_theta(theta):
    """
    Parameterization:
      theta = [a, b, g0, h]

      pi0    = logistic(a)           in (0,1)
      delta1 = exp(b)                >= 0
      sigma0 = exp(g0)               > 0
      sigma1 = sigma0 + exp(h)       >= sigma0

    This enforces sigma_1 >= sigma_0 automatically.
    """
    a, b, g0, h = theta

    pi0 = logistic(a)
    delta1 = np.exp(b)
    sigma0 = np.exp(g0)
    sigma1 = sigma0 + np.exp(h)

    return pi0, delta1, sigma0, sigma1


def neg_loglik(theta, K, x):
    """
    Negative log-likelihood for the maximum-of-mixtures model
    with unequal variances and sigma_1 >= sigma_0.
    """
    pi0, delta1, sigma0, sigma1 = unpack_theta(theta)

    z0 = x / sigma0
    z1 = (x - delta1) / sigma1

    F0 = ndtr(z0)
    F1 = ndtr(z1)

    f0 = np.exp(-0.5 * z0**2) / np.sqrt(2.0 * np.pi) / sigma0
    f1 = np.exp(-0.5 * z1**2) / np.sqrt(2.0 * np.pi) / sigma1

    mix_cdf = pi0 * F0 + (1.0 - pi0) * F1
    mix_pdf = pi0 * f0 + (1.0 - pi0) * f1

    dens = K * (mix_cdf ** (K - 1)) * mix_pdf

    if np.any(dens <= 0) or np.any(~np.isfinite(dens)):
        return 1e100

    return -np.sum(np.log(dens))


# ------------------------------------------------------------
# 5. ROBUSTIFIED OPTIMIZATION
# ------------------------------------------------------------

def get_bounds():
    """
    Bounds on transformed parameters [a, b, g0, h].

    These are numerical bounds, not identifying restrictions.
    They reduce the risk of unstable near-singular solutions.
    """
    return [
        (-10.0, 10.0),                          # a -> pi0
        (np.log(1e-10), np.log(2.0)),           # b -> delta1 >= 0
        (np.log(SIGMA0_MIN), np.log(2.0)),      # g0 -> sigma0
        (np.log(SIGMA_EXTRA_MIN), np.log(2.0)), # h -> sigma1 - sigma0
    ]


def initial_points():
    """
    Deterministic local multistart grid.
    """
    pts = []
    for pi0_0 in [0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95]:
        for delta1_0 in [1e-6, 0.01, 0.03, 0.05, 0.10, 0.20]:
            for sigma0_0 in [0.02, 0.05, 0.10, 0.20]:
                for extra_0 in [0.001, 0.01, 0.03, 0.05, 0.10]:
                    a0 = np.log(pi0_0 / (1.0 - pi0_0))
                    b0 = np.log(delta1_0 if delta1_0 > 0 else 1e-10)
                    g00 = np.log(sigma0_0)
                    h0 = np.log(extra_0)
                    pts.append(np.array([a0, b0, g00, h0], dtype=float))
    return pts


def local_refine(obj, x0, bounds):
    """
    One bounded local optimization.
    """
    return minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": LOCAL_MAXITER},
    )


def fit_for_K(K, x):
    """
    Strong optimizer:
      1) bounded local multistart (L-BFGS-B)
      2) global search (differential_evolution)
      3) final local polishing (L-BFGS-B)

    Parallel policy:
      - by default this function is serial
      - K-level multiprocessing is handled outside
    """
    bounds = get_bounds()
    obj = lambda th: neg_loglik(th, K, x)

    best_fun = np.inf
    best_x = None

    # Stage 1: deterministic local multistart
    for theta0 in initial_points():
        res = local_refine(obj, theta0, bounds)
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x

    # Stage 2: global search + local polish
    for base_seed in DE_SEEDS:
        seed = base_seed + int(K)

        de_res = differential_evolution(
            obj,
            bounds=bounds,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            seed=seed,
            polish=False,
            updating="deferred",
            workers=DE_WORKERS,
        )

        res = local_refine(obj, de_res.x, bounds)
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x

    # Stage 3: final polish from best point found
    final_res = local_refine(obj, best_x, bounds)
    return final_res


# ------------------------------------------------------------
# 6. FAMILYWISE ERROR RATES AND FDR
# ------------------------------------------------------------

def compute_table_row(K, x, c_vec):
    """
    Fit the model for fixed K, then compute:
      pi0, delta1, sigma0, sigma1, alpha_bar_K, beta_bar_K, log-likelihood, FDR
    """
    res = fit_for_K(K, x)
    pi0, delta1, sigma0, sigma1 = unpack_theta(res.x)

    # Primitive error rates per predictor
    alpha_n = 1.0 - ndtr(c_vec / sigma0)
    beta_n = ndtr((c_vec - delta1) / sigma1)

    # Familywise Type I error
    alphaK_n = 1.0 - (1.0 - alpha_n) ** K

    # Familywise Type II error
    betaK_n = (
        (pi0 * (1.0 - alpha_n) + (1.0 - pi0) * beta_n) ** K
        - (pi0 * (1.0 - alpha_n)) ** K
    ) / (1.0 - pi0 ** K)

    alpha_bar_K = alphaK_n.mean()
    beta_bar_K = betaK_n.mean()

    fdr = (alpha_bar_K * pi0) / (
        alpha_bar_K * pi0 + (1.0 - beta_bar_K) * (1.0 - pi0)
    )

    loglik = -res.fun

    return {
        "K": K,
        "pi_0": pi0,
        "delta_1": delta1,
        "sigma_0": sigma0,
        "sigma_1": sigma1,
        "alpha_K": alpha_bar_K,
        "beta_K": beta_bar_K,
        "log_likelihood": loglik,
        "FDR": fdr,
    }


def compute_table_row_worker(args):
    K, x, c_vec = args
    row = compute_table_row(K, x, c_vec)
    return K, row


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():
    stats_df, x_obs, c_vec = build_cross_section(CSV_PATH)

    print("Number of predictors:", len(stats_df))
    print("Mean rho:", stats_df["rho"].mean())
    print("Median rho:", stats_df["rho"].median())
    print("Mean AR(1) threshold:", stats_df["c_n"].mean())
    print(
        "Mean i.i.d.-Normal threshold:",
        (norm.ppf(1 - ALPHA / 2) / np.sqrt(stats_df["T"])).mean(),
    )
    print()

    table2_rows = []

    if PARALLEL_OVER_K:
        n_workers = max(1, int(MAX_WORKERS_K))
        print(f"Running in parallel across K with {n_workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(compute_table_row_worker, (K, x_obs, c_vec)): K
                for K in K_GRID
            }
            results_by_k = {}
            for fut in as_completed(futures):
                K, row = fut.result()
                results_by_k[K] = row
                print(f"Finished K={K}", flush=True)

        for K in sorted(results_by_k):
            table2_rows.append(results_by_k[K])
    else:
        print("Running serially across K...")
        for K in K_GRID:
            row = compute_table_row(K, x_obs, c_vec)
            table2_rows.append(row)
            print(f"Finished K={K}", flush=True)

    table2 = pd.DataFrame(table2_rows).sort_values("K").reset_index(drop=True)

    table2_display = table2.copy()
    for col in [
        "pi_0", "delta_1", "sigma_0", "sigma_1",
        "alpha_K", "beta_K", "log_likelihood", "FDR"
    ]:
        table2_display[col] = table2_display[col].round(6)

    print("\nTable 2 (monthly units, sigma_1 >= sigma_0, strong optimizer, parallel over K):")
    print(table2_display.to_string(index=False))

    table2_display.to_csv("Table2_sigma1_ge_sigma0_monthly_strongopt_parallel.csv", index=False)
    stats_df.to_csv("Section6_predictor_stats_monthly.csv", index=False)

    print("\nSaved:")
    print("  - Table2_sigma1_ge_sigma0_monthly_strongopt_parallel.csv")
    print("  - Section6_predictor_stats_monthly.csv")


if __name__ == "__main__":
    main()