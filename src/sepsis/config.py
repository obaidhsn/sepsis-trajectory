"""Tiny YAML config loader with `base:` inheritance and CLI overrides.

Example:
    cfg = load_config("configs/experiments/gat_transformer.yaml",
                      overrides=["train.seed=1", "model.hidden_dim=96"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(text: str) -> Any:
    value = yaml.safe_load(text)
    # PyYAML follows YAML 1.1, which reads "3e-4" (no dot) as a string.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_path = (path.parent / cfg.pop("base")).resolve()
        cfg = _deep_update(load_config(base_path), cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg
