from __future__ import annotations
from typing import Dict, Optional, Type
import numpy as np

from ...core import FloatingPointPrecision as F, PrecisionLike, NPFloat

_real_dtype_table = {
    F.SINGLE: np.single,
    F.DOUBLE: np.double,
    F.LONGDOUBLE: np.longdouble
}


_complex_dtype_table = {
    F.SINGLE: np.csingle,
    F.DOUBLE: np.cdouble,
    F.LONGDOUBLE: np.clongdouble
}


_nbytes_table = {
    F.SINGLE: 4,
    F.DOUBLE: 8,
    F.LONGDOUBLE: 16
}


_tol_table = {
    F.SINGLE: np.single(1e-6),
    F.DOUBLE: np.double(1e-12),
    F.LONGDOUBLE: np.longdouble(1e-18),
}


def real_dtype(p: PrecisionLike) -> Type[np.floating]:
    return _real_dtype_table[F.cast(p)]

def complex_dtype(p: PrecisionLike) -> Type[np.complexfloating]:
    return _complex_dtype_table[F.cast(p)]

def nbytes(p: PrecisionLike) -> int:
    return _nbytes_table[F.cast(p)]


def floating_tol(p: PrecisionLike, *, cfg: Optional[dict]=None) -> NPFloat:
    P = F.cast(p)
    if cfg is None:
        return _tol_table[P]
    return real_dtype(P)(cfg[P.name.lower()])


