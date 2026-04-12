import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# Table 1: Conditional upper-tail probabilities in Section 4.4.3
# ============================================================
#
# Case A (search and selection):
#   Trial-level mixture:
#       SR_hat_k ~ pi0_A * N(0,1) + (1-pi0_A) * N(delta1,1)
#   pi0_A = 0.95
#   K_A   = 10
#   reported statistic = max_{1<=k<=K_A} SR_hat_k
#
#   Therefore, the CDF of the reported statistic is
#       G_A(x) = [ pi0_A * Phi(x) + (1-pi0_A) * Phi(x-delta1) ]^K_A
#
# Case B (no search):
#   Trial-level mixture:
#       SR_hat ~ pi0_B * N(0,1) + (1-pi0_B) * N(delta1,1)
#   pi0_B = 0.15
#   K_B   = 1
#
#   Therefore, the CDF of the reported statistic is
#       G_B(x) = pi0_B * Phi(x) + (1-pi0_B) * Phi(x-delta1)
#
# Threshold:
#   c = 1.96
#
# The table reports:
#   Case_A(x) = P[X_A >= x | X_A >= c]
#   Case_B(x) = P[X_B >= x | X_B >= c]
# ============================================================

# Parameters consistent with revised Figures 2-4
pi0_A = 0.95
pi0_B = 0.15
delta1 = 0.30
K_A = 10
c = 1.96

# Grid used in Table 1
x_grid = np.arange(2.00, 4.01, 0.25)


# ---------- CDFs of reported statistics ----------
def cdf_case_A(x, pi0=pi0_A, delta=delta1, K=K_A):
    """
    CDF of the reported statistic in Case A:
        G_A(x) = [ pi0 * Phi(x) + (1-pi0) * Phi(x-delta) ]^K
    """
    mixture_cdf = pi0 * norm.cdf(x) + (1.0 - pi0) * norm.cdf(x - delta)
    return mixture_cdf ** K


def cdf_case_B(x, pi0=pi0_B, delta=delta1):
    """
    CDF of the reported statistic in Case B (no search):
        G_B(x) = pi0 * Phi(x) + (1-pi0) * Phi(x-delta)
    """
    return pi0 * norm.cdf(x) + (1.0 - pi0) * norm.cdf(x - delta)


# ---------- Unconditional upper-tail probabilities ----------
def tail_case_A(x, pi0=pi0_A, delta=delta1, K=K_A):
    return 1.0 - cdf_case_A(x, pi0, delta, K)


def tail_case_B(x, pi0=pi0_B, delta=delta1):
    return 1.0 - cdf_case_B(x, pi0, delta)


# ---------- Conditional upper-tail probabilities ----------
def cond_tail_case_A(x, c=c, pi0=pi0_A, delta=delta1, K=K_A):
    """
    P(X_A >= x | X_A >= c)
    """
    return tail_case_A(x, pi0, delta, K) / tail_case_A(c, pi0, delta, K)


def cond_tail_case_B(x, c=c, pi0=pi0_B, delta=delta1):
    """
    P(X_B >= x | X_B >= c)
    """
    return tail_case_B(x, pi0, delta) / tail_case_B(c, pi0, delta)


# ---------- Build Table 1 ----------
table = pd.DataFrame({
    "x": x_grid,
    "Case_A": [cond_tail_case_A(x) for x in x_grid],
    "Case_B": [cond_tail_case_B(x) for x in x_grid],
})

table["Diff"] = np.abs(table["Case_A"] - table["Case_B"])


print(f"\nmax(abs(diff)) = {table['Diff'].max():.6f}")

# Save both rounded and unrounded versions
table.to_csv("table 1.csv", index=False)
