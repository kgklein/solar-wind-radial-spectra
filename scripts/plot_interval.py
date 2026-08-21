#!/usr/bin/env python3
"""Run the solarwind diagnostic plotting command."""

import sys
from pathlib import Path

# A checkout uses the conventional src/ layout. Add it for this convenience
# script so the README command works before an editable package installation.
repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root / "src"))

try:
    from solarwind.plot import main
except ModuleNotFoundError as error:
    if error.name != "solarwind":
        raise SystemExit(
            f"Missing Python dependency '{error.name}'. Install the project first with "
            "python -m pip install -e '.[test]'"
        ) from error
    raise


if __name__ == "__main__":
    main()
