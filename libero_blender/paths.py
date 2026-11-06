"""Paths from configs/paths.yaml, or from the file named by $LIBERO_BLENDER_PATHS.

Importing this also points LIBERO at its own files and at `libero_datasets`, so it must be imported
before libero: LIBERO reads its config when it is first imported.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_paths = yaml.safe_load(Path(os.environ.get("LIBERO_BLENDER_PATHS", ROOT / "configs" / "paths.yaml")).read_text())
LIBERO_DATASETS = ROOT / _paths["libero_datasets"]
RENDERED_DATASETS = ROOT / _paths["rendered_datasets"]
RESULTS = ROOT / _paths["results"]
BLENDER = str(ROOT / _paths["blender"]) if "/" in _paths["blender"] else _paths["blender"]  # path or command


def _write_libero_config():
    libero = ROOT / "third_party" / "LIBERO" / "libero" / "libero"
    config = {"benchmark_root": libero, "bddl_files": libero / "bddl_files", "init_states": libero / "init_files",
              "assets": libero / "assets", "datasets": LIBERO_DATASETS}
    config_dir = ROOT / ".cache" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed, as parallel jobs start at the same time.
    temporary = config_dir / f"config.yaml.{os.getpid()}"
    temporary.write_text(yaml.safe_dump({key: str(path) for key, path in config.items()}))
    temporary.replace(config_dir / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


_write_libero_config()
