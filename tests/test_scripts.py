"""
Tests that run both replication scripts via `uv run` and assert they exit cleanly.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(script_name: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", script_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def test_code_table_1():
    result = run_script("code_table_1.py")
    assert result.returncode == 0, (
        f"code_table_1.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_code_table_2():
    result = run_script("code_table_2-robust-parallel.py")
    assert result.returncode == 0, (
        f"code_table_2-robust-parallel.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
