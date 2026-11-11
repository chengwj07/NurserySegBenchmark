"""YOLO26 + SAM2 semantic segmentation project.

Environment variables are automatically set when importing this package to resolve OpenMP conflicts and
Ultralytics configuration directory permission issue.
"""
import os
import pathlib

# ---- Resolve OpenMP repeated loading conflicts (torch + numpy) ----
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- Redirect the Ultralytics configuration directory to the project to avoid failure to write AppData ----
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_cfg_dir = _PROJECT_ROOT / ".ultralytics_cfg"
_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cfg_dir))

__version__ = "1.0.0"
