import os
import subprocess
import sys
from pathlib import Path


def test_checkout_plot_script_finds_src_package_without_pythonpath():
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/plot_interval.py", "--help"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--mission" in result.stdout
