import json
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)
