from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible greedy inference as much as possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This can make some operations deterministic, but may slightly reduce speed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
