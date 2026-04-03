# FDR in Finance — Replication Code

This repository contains the replication scripts for the paper's tables and simulation results.

## Scripts

| Script | Description |
|--------|-------------|
| `code_table_1.py` | Reproduces Table 1: conditional tail probabilities (Section 4.4.3) |
| `code_table_2-robust-parallel.py` | Reproduces Table 2: FDR estimates with robust, parallelized MLE (Section 6) |

## Running with `uv` (recommended)

Both scripts embed [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so [`uv`](https://docs.astral.sh/uv/) can provision an isolated, reproducible environment automatically — no manual `pip install` or virtual environment setup required.

### Install `uv`

Follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via `pip`:

```bash
pip install uv
```

### Run the scripts

```bash
uv run code_table_1.py
uv run code_table_2-robust-parallel.py
```

`uv` will create an isolated virtual environment, install the pinned dependencies (`numpy==2.2.6`, `pandas==2.2.3`, `scipy==1.15.3`), and execute the script. The environment is cached and reused on subsequent runs.

## Data

`code_table_2-robust-parallel.py` requires `PredictorLSretWide.csv` to be present in the working directory.

## Output files

| Script | Output |
|--------|--------|
| `code_table_1.py` | `table1_conditional_tail_probabilities.csv`, `table1_conditional_tail_probabilities_unrounded.csv` |
| `code_table_2-robust-parallel.py` | `Table2_sigma1_ge_sigma0_monthly_strongopt_parallel.csv`, `Section6_predictor_stats_monthly.csv` |
