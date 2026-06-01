"""Compatibility module for the maintained GNN transformer planner path."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BACKUP_PATH = Path(__file__).with_name("gnn_transformer_planner_backup(backup).py")
_SPEC = spec_from_file_location("GNN_Trans.planner._gnn_transformer_planner_backup", _BACKUP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load planner backup module from {_BACKUP_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

GNNTransformerPlanner = _MODULE.GNNTransformerPlanner

__all__ = ["GNNTransformerPlanner"]
