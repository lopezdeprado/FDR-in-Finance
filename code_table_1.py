# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy==2.2.6",
#   "pandas==2.2.3",
#   "scipy==1.15.3",
# ]
# ///

import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# Table 1: Conditional tail probabilities in Section 4.4.3
# ============================================================
#
# Case A (search and selection, NEW FRAMEWORK):
#   Each latent trial is drawn from the trial-level mixture
#       SR_hat_k ~ pi0 * N(0,1) + (1-pi0) * N(SR1,1)
#   pi0 = 0.75
#   K   = 5
#   selected statistic = max_{1<=k<=K} SR_hat_k
#
#   Therefore, the CDF of the selected statistic is
#       F_A(x) = [ pi0 * Phi(x) + (1-pi0) * Phi(x-SR1) ]^K
#
# Case B (no search):
#   H0: SR_hat ~ N(0, 1)
#   H1: SR_hat ~ N(SR1, 1)
#   pi0' = 0.10
#   K'  = 1
#
#   Therefore, the CDF is
#       F_B(x) = pi0' * Phi(x) + (1-pi0') * Phi(x-SR1)
#
# Threshold:
#   c = 1.96
#
# The table reports:
#   Case_A(x) = P[X_A >= x | X_A >= c]
#   Case_B(x) = P[X_B >= x | X_B >= c]
# ============================================================

# Parameters
pi0_A = 0.75
pi0_B = 0.10
SR1 = 0.30
K = 5
c = 1.96

# Grid used in Table 1
x_grid = np.arange(2.00, 4.01, 0.25)

# ---------- CDFs under the new framework ----------

def cdf_case_A(x, pi0=pi0_A, sr1=SR1, K=K):
    """
    CDF of the selected statistic in Case A under the NEW framework:
        F_A(x) = [ pi0 * Phi(x) + (1-pi0) * Phi(x-sr1) ]^K
    """
    mixture_cdf = pi0 * norm.cdf(x) + (1.0 - pi0) * norm.cdf(x - sr1)
    return mixture_cdf ** K

def cdf_case_B(x, pi0=pi0_B, sr1=SR1):
    """
    CDF of the reported statistic in Case B:
        F_B(x) = pi0 * Phi(x) + (1-pi0) * Phi(x-sr1)
    """
    return pi0 * norm.cdf(x) + (1.0 - pi0) * norm.cdf(x - sr1)

# ---------- Unconditional tail probabilities ----------

def tail_case_A(x, pi0=pi0_A, sr1=SR1, K=K):
    """
    Unconditional tail probability in Case A:
        P(X_A >= x) = 1 - F_A(x)
    """
    return 1.0 - cdf_case_A(x, pi0, sr1, K)

def tail_case_B(x, pi0=pi0_B, sr1=SR1):
    """
    Unconditional tail probability in Case B:
        P(X_B >= x) = 1 - F_B(x)
    """
    return 1.0 - cdf_case_B(x, pi0, sr1)

# ---------- Conditional tail probabilities ----------

def cond_tail_case_A(x, c=c, pi0=pi0_A, sr1=SR1, K=K):
    """
    P(X_A >= x | X_A >= c)
    """
    return tail_case_A(x, pi0, sr1, K) / tail_case_A(c, pi0, sr1, K)

def cond_tail_case_B(x, c=c, pi0=pi0_B, sr1=SR1):
    """
    P(X_B >= x | X_B >= c)
    """
    return tail_case_B(x, pi0, sr1) / tail_case_B(c, pi0, sr1)

# ---------- Build Table 1 ----------

table = pd.DataFrame({
    "x": x_grid,
    "Case_A": [cond_tail_case_A(x) for x in x_grid],
    "Case_B": [cond_tail_case_B(x) for x in x_grid],
})

table["Diff"] = np.abs(table["Case_A"] - table["Case_B"])

# Rounded version for display
table_rounded = table.copy()
for col in ["x", "Case_A", "Case_B", "Diff"]:
    table_rounded[col] = table_rounded[col].round(3)

print(table_rounded.to_string(index=False))

# Optional: save both rounded and unrounded versions
table.to_csv("table1_conditional_tail_probabilities_unrounded.csv", index=False)
table_rounded.to_csv("table1_conditional_tail_probabilities.csv", index=False)