"""Configure one predictable on-disk cache root for PySPEDAS."""

import os
import sys
from pathlib import Path


def default_data_directory() -> Path:
    """Return this checkout's data directory, or data/ in the working directory."""

    checkout_root = Path(__file__).resolve().parents[2]
    if (checkout_root / "pyproject.toml").is_file():
        return checkout_root / "data"
    return Path.cwd() / "data"


def configure_pyspedas_cache() -> Path:
    """Set the shared cache root before mission-specific PySPEDAS imports."""

    data_directory = Path(
        os.environ.setdefault("SPEDAS_DATA_DIR", str(default_data_directory()))
    ).expanduser().resolve()

    # PySPEDAS reads its environment configuration at import time. Refresh an
    # already-imported config too, while respecting explicit mission overrides.
    loaded_configs = (
        ("pyspedas.projects.psp.config", "PSP_DATA_DIR", "psp"),
        ("pyspedas.projects.solo.config", "SOLO_DATA_DIR", "solar-orbiter"),
    )
    for module_name, override_name, mission_directory in loaded_configs:
        module = sys.modules.get(module_name)
        if module is not None and override_name not in os.environ:
            module.CONFIG["local_data_dir"] = str(data_directory / mission_directory)

    return data_directory
