from __future__ import annotations
from typing import Dict, Literal, Type
import numpy as np
from enum import Enum


class FloatingPointPrecision(Enum):
    SINGLE = 'SINGLE'
    DOUBLE = 'DOUBLE'
    LONGDOUBLE = 'LONGDOUBLE'

    def __str__(self) -> str:
        return self.name


PrecisionStr = Literal[
    'SINGLE', 'DOUBLE', 'LONGDOUBLE',
    'single', 'double', 'longdouble',
]

_real_dtype = {
    'SINGLE': np.single,
    'DOUBLE': np.double,
    'LONGDOUBLE': np.longdouble,
}

_complex_dtype = {
    'SINGLE': np.csingle,
    'DOUBLE': np.cdouble,
    'LONGDOUBLE': np.clongdouble,
}


def get_real_dtype(
        precision: FloatingPointPrecision | PrecisionStr
    ) -> Type[np.floating]:
    return _real_dtype[str(precision).upper()]


def get_complex_dtype(
        precision: FloatingPointPrecision | PrecisionStr
    ) -> Type[np.complexfloating]:
    return _complex_dtype[str(precision).upper()]

