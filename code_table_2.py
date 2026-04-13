# ============================================================
# Replication code for Tables 2 & 3
# Unified version: choose between
#   STAT_MODE = "raw"     -> fit on raw monthly Sharpe ratios
#   STAT_MODE = "pivotal" -> fit on pivotalized statistic
#
# IMPORTANT:
#   In BOTH modes, sigma_0 and sigma_1 are estimated.
#   The only difference between modes is:
#     - the cross-sectional statistic x_obs
#     - the rejection thresholds c_n
#
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
K_GRID = list(range(1, 11)) + [25, 50, 75, 100]

# SINGLE SWITCH:
#   "raw"     -> original implementation in raw monthly Sharpe-ratio space
#   "pivotal" -> pivotalized statistic:
#                sqrt(T_n * (1-rho_n)/(1+rho_n)) * SR_hat_n
STAT_MODE = "raw"   # change only this line if needed

# Lower bound on delta_1.
# Set DELTA1_MIN = 0.10 for "raw" and 2.0 for "pivotal", to require economically meaningful positive separation.
DELTA1_MIN = 0.10

# Optimizer settings
DE_MAXITER = 40
DE_POPSIZE = 12
DE_SEEDS = [101, 202, 303]
LOCAL_MAXITER = 5000

# Lower bounds on scale parameters.
# These are numerical safeguards, not identifying restrictions.
SIGMA0_MIN = 0.02
SIGMA_EXTRA_MIN = 0.001

# Parallel settings
PARALLEL_OVER_K = True
MAX_WORKERS_K = min(len(list(K_GRID)), os.cpu_count() or 1)
DE_WORKERS = 1  # set to -1 only if PARALLEL_OVER_K = False


# ------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------
def raw_sharpe(x: pd.Series) -> float:
    """
    Raw (monthly, non-annualized) sample Sharpe ratio:
        SR = mean / std
    """
    x = x.dropna().astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)
    return mu / sd



def ar1_autocorr(x: pd.Series) -> float:
    """First-order sample autocorrelation."""
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



def pivotalize_sharpe(sr: float, T: int, rho: float) -> float:
    """
    Pivotalized Sharpe statistic:
        z = sqrt(T * (1-rho)/(1+rho)) * SR_hat
    """
    return np.sqrt(T * (1.0 - rho) / (1.0 + rho)) * sr


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

        row = {
            "predictor": name,
            "T": T,
            "rho": rho,
            "SR_hat": sr,
        }

        if STAT_MODE == "raw":
            c = ar1_threshold(T=T, rho=rho, alpha=ALPHA)
            row["x_obs"] = sr
            row["c_n"] = c

        elif STAT_MODE == "pivotal":
            z = pivotalize_sharpe(sr=sr, T=T, rho=rho)
            zcrit = norm.ppf(1 - ALPHA / 2)
            row["x_obs"] = z
            row["c_n"] = zcrit

        else:
            raise ValueError(f"Unknown STAT_MODE={STAT_MODE}")

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    x_obs = stats_df["x_obs"].to_numpy(dtype=float)
    c_vec = stats_df["c_n"].to_numpy(dtype=float)
    return stats_df, x_obs, c_vec


# ------------------------------------------------------------
# 4. MAXIMUM-OF-MIXTURES LIKELIHOOD (TRIAL LEVEL)
# ------------------------------------------------------------


def logistic(a: float) -> float:
    """Map R to (0,1)."""
    return 1.0 / (1.0 + np.exp(-a))



def unpack_theta(theta):
    """
    Parameterization (same in BOTH modes):
      theta = [a, b, g0, h]

      pi0_trial = logistic(a)         in (0,1)
      delta1    = exp(b)              >= DELTA1_MIN
      sigma0    = exp(g0)             > 0
      sigma1    = sigma0 + exp(h)     >= sigma0

    This enforces sigma_1 >= sigma_0 automatically.
    The lower bound on delta_1 is imposed through the optimizer bounds.
    """
    a, b, g0, h = theta

    pi0_trial = logistic(a)
    delta1 = np.exp(b)
    sigma0 = np.exp(g0)
    sigma1 = sigma0 + np.exp(h)

    return pi0_trial, delta1, sigma0, sigma1



def neg_loglik(theta, K, x):
    """
    Negative log-likelihood for the maximum-of-mixtures model
    with unequal variances and sigma_1 >= sigma_0.

    IMPORTANT: this is a trial-level likelihood.
    pi0_trial is the probability that one candidate trial is null.
    """
    pi0_trial, delta1, sigma0, sigma1 = unpack_theta(theta)

    z0 = x / sigma0
    z1 = (x - delta1) / sigma1

    F0 = ndtr(z0)
    F1 = ndtr(z1)

    f0 = np.exp(-0.5 * z0**2) / np.sqrt(2.0 * np.pi) / sigma0
    f1 = np.exp(-0.5 * z1**2) / np.sqrt(2.0 * np.pi) / sigma1

    mix_cdf = pi0_trial * F0 + (1.0 - pi0_trial) * F1
    mix_pdf = pi0_trial * f0 + (1.0 - pi0_trial) * f1

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

    These are numerical bounds, not identifying restrictions,
    except for DELTA1_MIN, which is a user-imposed lower bound.
    """
    sigma_upper = 2.0 if STAT_MODE == "raw" else 20.0

    return [
        (-10.0, 10.0),                                   # a -> pi0_trial
        (np.log(DELTA1_MIN), np.log(20.0)),              # b -> delta1 >= DELTA1_MIN
        (np.log(SIGMA0_MIN), np.log(sigma_upper)),       # g0 -> sigma0
        (np.log(SIGMA_EXTRA_MIN), np.log(sigma_upper)),  # h -> sigma1 - sigma0
    ]



def initial_points():
    """Deterministic local multistart grid."""
    pts = []

    if STAT_MODE == "raw":
        delta_grid_base = [0.10, 0.20, 0.50, 1.0]
        sigma0_grid = [0.02, 0.05, 0.10, 0.20]
        extra_grid = [0.001, 0.01, 0.03, 0.05, 0.10]
    else:
        delta_grid_base = [0.10, 0.20, 0.50, 1.0, 2.0]
        sigma0_grid = [0.20, 0.50, 1.00, 2.00, 4.00]
        extra_grid = [0.001, 0.01, 0.03, 0.05, 0.10, 0.50, 1.00, 2.00]

    delta_grid = sorted(set([DELTA1_MIN] + [d for d in delta_grid_base if d >= DELTA1_MIN]))

    for pi0_0 in [0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95]:
        for delta1_0 in delta_grid:
            for sigma0_0 in sigma0_grid:
                for extra_0 in extra_grid:
                    a0 = np.log(pi0_0 / (1.0 - pi0_0))
                    b0 = np.log(delta1_0)
                    g00 = np.log(sigma0_0)
                    h0 = np.log(extra_0)
                    pts.append(np.array([a0, b0, g00, h0], dtype=float))

    return pts



def local_refine(obj, x0, bounds):
    """One bounded local optimization."""
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
    """
    bounds = get_bounds()
    obj = lambda th: neg_loglik(th, K, x)

    best_fun = np.inf
    best_x = None

    for theta0 in initial_points():
        res = local_refine(obj, theta0, bounds)
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x

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

    final_res = local_refine(obj, best_x, bounds)
    return final_res


# ------------------------------------------------------------
# 6. FAMILYWISE ERROR RATES AND SEARCH-ADJUSTED FDR
# ------------------------------------------------------------


def compute_table_row(K, x, c_vec):
    """
    Fit the model for fixed K, then compute:
      - trial-level primitive parameters: pi_0_trial, delta_1, sigma_0, sigma_1
      - familywise/search-adjusted error rates: alpha_K, beta_K
      - reported-object null prior: pi_0_reported = P(M=0) = pi_0_trial ** K
      - search-adjusted FDR
    """
    res = fit_for_K(K, x)
    pi0_trial, delta1, sigma0, sigma1 = unpack_theta(res.x)

    # Primitive single-trial error rates per predictor.
    alpha_n = 1.0 - ndtr(c_vec / sigma0)
    beta_n = ndtr((c_vec - delta1) / sigma1)

    # Familywise / search-adjusted Type I error:
    alphaK_n = 1.0 - (1.0 - alpha_n) ** K

    # Familywise / search-adjusted Type II error:
    betaK_n = (
        (pi0_trial * (1.0 - alpha_n) + (1.0 - pi0_trial) * beta_n) ** K
        - (pi0_trial * (1.0 - alpha_n)) ** K
    ) / (1.0 - pi0_trial ** K)

    alpha_bar_K = alphaK_n.mean()
    beta_bar_K = betaK_n.mean()
    pi0_reported = pi0_trial ** K

    fdr_search = (alpha_bar_K * pi0_reported) / (
        alpha_bar_K * pi0_reported + (1.0 - beta_bar_K) * (1.0 - pi0_reported)
    )

    loglik = -res.fun

    return {
        "K": K,
        "pi_0": pi0_trial,
        "delta_1": delta1,
        "sigma_0": sigma0,
        "sigma_1": sigma1,
        "pi_0_K": pi0_reported,
        "alpha_K": alpha_bar_K,
        "beta_K": beta_bar_K,
        "log_likelihood": loglik,
        "FDR_search": fdr_search,
    }



def compute_table_row_worker(args):
    K, x, c_vec = args
    row = compute_table_row(K, x, c_vec)
    return K, row


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------


def main():
    print(f"Running mode: {STAT_MODE}")
    print(f"delta_1 lower bound: {DELTA1_MIN}")
    stats_df, x_obs, c_vec = build_cross_section(CSV_PATH)

    print("Number of predictors:", len(stats_df))
    print("Mean rho:", stats_df["rho"].mean())
    print("Median rho:", stats_df["rho"].median())

    if STAT_MODE == "raw":
        print("Mean AR(1) threshold:", stats_df["c_n"].mean())
        print(
            "Mean i.i.d.-Normal threshold:",
            (norm.ppf(1 - ALPHA / 2) / np.sqrt(stats_df["T"])).mean(),
        )
    elif STAT_MODE == "pivotal":
        print("Common pivotal threshold:", c_vec[0])
        print("Mean pivotalized statistic:", stats_df["x_obs"].mean())
        print("Std. dev. pivotalized statistic:", stats_df["x_obs"].std(ddof=1))

    print()

    table_rows = []

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
            table_rows.append(results_by_k[K])
    else:
        print("Running serially across K...")
        for K in K_GRID:
            row = compute_table_row(K, x_obs, c_vec)
            table_rows.append(row)
            print(f"Finished K={K}", flush=True)

    table = pd.DataFrame(table_rows).sort_values("K").reset_index(drop=True)

    table_display = table.copy()
    for col in [
        "pi_0", "pi_0_K", "delta_1", "sigma_0", "sigma_1",
        "alpha_K", "beta_K", "log_likelihood", "FDR_search"
    ]:
        table_display[col] = table_display[col].round(6)

    print(f"\nSection 6 / Appendix 4 table ({STAT_MODE} mode):")
    print(table_display.to_string(index=False))

    suffix = "raw" if STAT_MODE == "raw" else "pivotal"
    table_name = f"Table_section6_{suffix}_delta1min_{DELTA1_MIN:.2f}.csv"
    stats_name = f"Section6_predictor_stats_{suffix}_delta1min_{DELTA1_MIN:.2f}.csv"

    table_display.to_csv(table_name, index=False)
    stats_df.to_csv(stats_name, index=False)

    print("\nSaved:")
    print(f"  - {table_name}")
    print(f"  - {stats_name}")


if __name__ == "__main__":
    main()
