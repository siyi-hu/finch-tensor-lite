import os
import shutil
import sys
import sysconfig
import tomllib
from pathlib import Path

import donfig

"""
Finch Configuration Module

This module manages configuration settings for the Finch application.
Finch stores its settings and data in the `FINCH_PATH` directory, which
defaults to `~/.finch` but can be customized using the `FINCH_PATH`
environment variable.

Configuration details:
- Settings are stored in a `config.json` file within the `FINCH_PATH` directory.
- Values can be set via environment variables, the `config.json` file,
    or the `set_config` function.
- Configuration values are loaded automatically when the module is imported
    and can be accessed using the `get_config` function.

Use this module to easily manage and retrieve Finch-specific settings.
"""

is_windows = os.name == "nt"
is_apple = sys.platform == "darwin"

default = {
    "data_path": str(Path(sysconfig.get_path("data")) / "finch"),
    "cache_size": 10_000,
    "cache_enable": True,
    "cc": (
        os.getenv("CC")
        or sysconfig.get_config_var("CC")
        or str(shutil.which("gcc") or "cl" if is_windows else "cc")
    ),
    "cflags": os.getenv(
        "CFLAGS",
        [
            "-shared",
            "-fPIC",
            "-O3",
        ],
    ),
    "shlib_suffix": (
        sysconfig.get_config_var("SHLIB_SUFFIX") or (".dll" if is_windows else ".so")
    ),
}

config = donfig.Config("finch", defaults=[default])


def get_version():
    """
    Get the version of Finch.
    """
    pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found.")

    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    try:
        return pyproject_data["project"]["version"]
    except KeyError:
        raise ValueError("Version not found in pyproject.toml.") from None
