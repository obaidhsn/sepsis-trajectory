"""Build any model from the config dict (see configs/default.yaml)."""

from __future__ import annotations

from ..constants import DYNAMIC, STATIC, VITALS
from .autoformer import Autoformer
from .baselines import LinearForecaster, MLPMultiTask, Persistence
from .gat_transformer import GATTransformer
from .grud import GRUD
from .informer import Informer
from .transformer import TemporalTransformer

MODEL_REGISTRY = {
    "persistence": Persistence,
    "linear": LinearForecaster,
    "mlp": MLPMultiTask,
    "grud": GRUD,
    "transformer": TemporalTransformer,
    "informer": Informer,
    "autoformer": Autoformer,
    "gat_transformer": GATTransformer,
}


def build_model(cfg: dict, adjacency=None):
    m = dict(cfg["model"])
    name = m.pop("name")
    dims = dict(
        n_vars=len(DYNAMIC),
        n_features=3 * len(DYNAMIC) + 1,
        n_static=len(STATIC),
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        n_vitals=len(VITALS),
        n_onset=len(cfg["data"]["onset_edges"]) + 1,
        tasks=tuple(cfg["train"]["tasks"]),
    )
    if name == "gat_transformer":
        if adjacency is None:
            raise ValueError("gat_transformer needs an adjacency matrix")
        dims["adjacency"] = adjacency
    m.pop("graph", None)
    return MODEL_REGISTRY[name](**dims, **m)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
