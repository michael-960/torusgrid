from __future__ import annotations
from numbers import Number
from typing import Any, overload
import numpy as np


@overload
def scalarize(data: list) -> list: ...

@overload
def scalarize(data: dict) -> dict: ...

@overload
def scalarize(data: np.ndarray) -> np.ndarray | Number | np.generic: ...

@overload
def scalarize(data: str) -> str: ...

@overload
def scalarize(data: Number | np.generic) -> Number | np.generic: ...

def scalarize(data: dict | list | np.generic | Number | str | np.ndarray) -> Any:
    """
    When an NPZ file is loaded with np.load(), scalar values are stored as 0d
    arrays. This function converts all 0d arrays in a given object into
    scalars.
    """

    if np.isscalar(data):
        return data

    if isinstance(data, np.ndarray):
        try:
            return scalarize(data.item())
        except ValueError:
            return data

    if isinstance(data, dict):
        r = dict()
        for key, value in data.items():
            r[key] = scalarize(value)
        return r
    
    if isinstance(data, list):
        return [scalarize(item) for item in data]



