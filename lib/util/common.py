import json
import os
import os.path as osp
import pickle
import re
from pathlib import Path
from typing import Any
import torch 
import random
import numpy as np

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists; creates it if it does not.
    """
    if not osp.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False