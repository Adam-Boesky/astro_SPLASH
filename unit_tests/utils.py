import numpy as np

from typing import Union


def is_close(a: Union[float, np.ndarray], b: Union[float, np.ndarray], thresh: float = 10E-7) -> bool:
    if isinstance(a, float):
        return np.abs(a - b) < thresh
    return np.all(np.abs(a - b) < thresh)
