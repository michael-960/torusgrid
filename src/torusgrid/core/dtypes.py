from __future__ import annotations
from typing import Any, Literal, Type, Union, Sequence
from typing_extensions import Self
import numpy as np
import numpy.typing as npt
from enum import Enum


class FloatingPointPrecision(Enum):
    SINGLE = 'SINGLE'
    DOUBLE = 'DOUBLE'
    LONGDOUBLE = 'LONGDOUBLE'

    def __str__(self) -> str:
        return self.name

    @classmethod
    def cast(cls, precision: PrecisionLike) -> Self:
        if isinstance(precision, FloatingPointPrecision):
            return precision


        precision_ = str(precision).upper()
        return {
            'SINGLE': cls.SINGLE,
            'DOUBLE': cls.DOUBLE,
            'LONGDOUBLE': cls.LONGDOUBLE,
        }[precision_]

    @classmethod
    def from_dtype(cls, dtype) -> Self:
        if dtype in [np.longdouble, np.clongdouble]:
            return cls.LONGDOUBLE

        if dtype in [np.double, np.cdouble]:
            return cls.DOUBLE

        if dtype in [np.single, np.csingle]:
            return cls.SINGLE

        raise ValueError(f'Unrecognized dtype: {dtype}')

    @classmethod
    def most_precise(cls, *precisions: PrecisionLike) -> Self:
        """
        Return the most precise precision among all
        """
        precisions = tuple(cls.cast(p) for p in precisions)
        
        for precision in [cls.LONGDOUBLE, cls.DOUBLE, cls.SINGLE]:
            if precision in precisions:
                return precision
        
        raise RuntimeError()



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

PrecisionLike = Union[FloatingPointPrecision, PrecisionStr]


def get_real_dtype(
        precision: PrecisionLike
    ) -> Type[np.floating]:
    return _real_dtype[str(precision).upper()]


def get_complex_dtype(
        precision: PrecisionLike
    ) -> Type[np.complexfloating]:
    return _complex_dtype[str(precision).upper()]


def get_dtype(precision: PrecisionLike, *, complex: bool=False):
    if complex:
        return get_complex_dtype(precision)
    else:
        return get_real_dtype(precision)


NPInt = Union[np.integer, np.int_]
IntLike = Union[NPInt, int]


NPFloat = Union[np.floating, np.float_]
'single, double, longdouble, float_'

FloatLike = Union[NPFloat, float]
'single, double, longdouble, float_, float'

NPComplex = Union[np.complexfloating, np.complex_]
'csingle, cdouble, clongdouble, complex_'

ComplexLike = Union[NPComplex, complex]
'csingle, cdouble, clongdouble, complex_, complex'


SizeLike = Union[Sequence[FloatLike], npt.NDArray[np.floating], npt.NDArray[np.float_]]



def is_real_scalar(x: Any) -> bool:
    return isinstance(x, (int,float,np.integer,np.floating))

def is_real_sequence(x: Any, length: int|None = None) -> bool:
    if np.isscalar(x): return False
    try:
        arr = np.array(x, dtype=np.float_)

        if arr.ndim != 1: return False
        if length is None: return True
        return len(arr) == length

    except ValueError:
        return False


def is_int_scalar(x: Any) -> bool:
    return isinstance(x, (int,np.integer))

def is_int_sequence(x: Any, length: int|None = None) -> bool:
    if np.isscalar(x): return False
    try:
        arr = np.array(x, dtype=np.int_)
        if arr.ndim != 1: return False

        if not np.array_equal(x, arr): return False

        if length is None: return True

        return len(arr) == length

    except ValueError:
        return False



def float_fmt(
    x: FloatLike, 
    digits: int, sign: bool = True,
):
    """
    Scientific notation of a float number
    """
    s = np.format_float_scientific(x, precision=digits, min_digits=digits, sign=sign)
    return s


def highlight_last_digits(
    formatted_float: str,
    digits: int,
    highlight=('(', ')')
):
    """
    """
    
    a, e = formatted_float.split('e')
    
    if isinstance(highlight, tuple):
        l = highlight[0]
        r = highlight[1]

    elif isinstance(highlight, str):
        l = f'[{highlight}]'
        r = f'[/{highlight}]'
    else:
        raise ValueError(f'Invalid highlight method: {highlight}')

    a = a[:-digits] + l + a[-digits:] + r

    return f'{a}e{e}'



