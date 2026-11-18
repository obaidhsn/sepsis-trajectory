import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuDNN autotuning picks kernels non-deterministically; turn it off.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
