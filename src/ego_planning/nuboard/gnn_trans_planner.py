"""Compatibility module for the maintained NuBoard planner import path."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BACKUP_PATH = Path(__file__).with_name("gnn_trans_planner(backup).py")
_SPEC = spec_from_file_location("GNN_Trans.Nuboard._gnn_trans_planner_backup", _BACKUP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load NuBoard planner backup module from {_BACKUP_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

GNNTransPlanner = _MODULE.GNNTransPlanner

__all__ = ["GNNTransPlanner"]
