from __future__ import annotations
import numpy as np

from .dtypes import PrecisionLike


PI_LONG = np.longdouble('3.1415926535897932384626433832795028841971693993')
PI_DOUBLE = np.double(PI_LONG)
PI_SINGLE = np.single(PI_LONG)


_pi = {
    'LONGDOUBLE': PI_LONG,
    'DOUBLE': PI_DOUBLE,
    'SINGLE': PI_SINGLE
}

def pi(precision: PrecisionLike) -> np.longdouble | np.double | np.single:
    return _pi[str(precision).upper()]


E_LONG = np.longdouble('2.718281828459045235360287471352662497757')
E_DOUBLE = np.double(E_LONG)
E_SINGLE = np.single(E_LONG)

_e = {
    'LONGDOUBLE': E_LONG,
    'DOUBLE': E_DOUBLE,
    'SINGLE': E_SINGLE
}

def e(precision: PrecisionLike) -> np.longdouble | np.double | np.single:
    return _e[str(precision).upper()]


