from __future__ import annotations
from typing import Literal, Type, Union, Sequence
import numpy as np
import numpy.typing as npt
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

PrecisionLike = Union[FloatingPointError, PrecisionStr]

def get_real_dtype(
        precision: PrecisionLike
    ) -> Type[np.floating]:
    return _real_dtype[str(precision).upper()]


def get_complex_dtype(
        precision: PrecisionLike
    ) -> Type[np.complexfloating]:
    return _complex_dtype[str(precision).upper()]



NPFloat = Union[np.floating, np.float_]
'single, double, longdouble, float_'

FloatLike = Union[NPFloat, float]
'single, double, longdouble, float_, float'

NPComplex = Union[np.complexfloating, np.complex_]
'csingle, cdouble, clongdouble, complex_'

ComplexLike = Union[NPComplex, complex]
'csingle, cdouble, clongdouble, complex_, complex'


SizeLike = Union[Sequence[FloatLike], npt.NDArray[np.floating], npt.NDArray[np.float_]]



