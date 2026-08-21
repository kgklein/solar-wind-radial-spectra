import sys
from pathlib import Path
from types import SimpleNamespace

from solarwind.cache import configure_pyspedas_cache, default_data_directory


def test_default_cache_is_repository_data_directory(monkeypatch):
    monkeypatch.delenv("SPEDAS_DATA_DIR", raising=False)

    configured = configure_pyspedas_cache()

    repository_root = Path(__file__).resolve().parents[1]
    assert configured == repository_root / "data"
    assert default_data_directory() == repository_root / "data"


def test_explicit_shared_cache_override_is_preserved(monkeypatch, tmp_path):
    custom_cache = tmp_path / "spacecraft-cache"
    monkeypatch.setenv("SPEDAS_DATA_DIR", str(custom_cache))

    configured = configure_pyspedas_cache()

    assert configured == custom_cache


def test_already_imported_pyspedas_configs_are_refreshed(monkeypatch):
    monkeypatch.delenv("SPEDAS_DATA_DIR", raising=False)
    psp_config = SimpleNamespace(CONFIG={"local_data_dir": "psp_data/"})
    solo_config = SimpleNamespace(CONFIG={"local_data_dir": "solar_orbiter_data/"})
    monkeypatch.setitem(sys.modules, "pyspedas.projects.psp.config", psp_config)
    monkeypatch.setitem(sys.modules, "pyspedas.projects.solo.config", solo_config)

    configured = configure_pyspedas_cache()

    assert psp_config.CONFIG["local_data_dir"] == str(configured / "psp")
    assert solo_config.CONFIG["local_data_dir"] == str(configured / "solar-orbiter")
