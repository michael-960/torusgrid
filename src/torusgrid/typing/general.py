from __future__ import annotations

from typing import Sequence, Union
import numpy as np
import numpy.typing as npt


NPFloat = Union[np.floating, np.float_]
'single, double, longdouble, float_'

FloatLike = Union[NPFloat, float]
'single, double, longdouble, float_, float'

NPComplex = Union[np.complexfloating, np.complex_]
'csingle, cdouble, clongdouble, complex_'

ComplexLike = Union[NPComplex, complex]
'csingle, cdouble, clongdouble, complex_, complex'


SizeLike = Union[Sequence[FloatLike], npt.NDArray[np.floating], npt.NDArray[np.float_]]




