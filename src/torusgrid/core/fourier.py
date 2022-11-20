from __future__ import annotations
from typing import Literal, Tuple, TypeAlias
import numpy as np
import numpy.typing as npt

import warnings

from .dtypes import PrecisionLike, FloatLike, get_real_dtype


FFTWEffort: TypeAlias = Literal['FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE']


def generate_xk(
    L: FloatLike, N: int, *,
    center: bool=False, real: bool=False,
    precision: PrecisionLike = 'double'
) ->    Tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], 
        np.floating, np.floating]:
    r"""
    Generate x and k for Fourier transform given system size (L) and number of
    samples (N)

    :param L: system size
    :param N: number of sample points
    :param center: whether to sample from :math:`(-L/2, L/2)` instead of :math:`(0, L)`
    :param real: if :code:`True`, the resulting k-space samples be :math:`(0, \Delta k, \dotsb, (N/2+1)\Delta k)`
    :param precision: precision of the generated arrays

    :return: :math:`x`, :math:`k`, :math:`dx`, :math:`dk`
    """
    dtype = get_real_dtype(precision)

    if not isinstance(L, dtype):
        warnings.warn(f'Precision mismatch: {dtype} and {type(L)}')
        L = dtype(L)

    dx = L / N
    dk = np.pi*2 / L

    M = N // 2

    if center:
        x = np.linspace(-L/2, L/2, N, endpoint=False, dtype=dtype)
    else:
        x = np.linspace(0, L, N, endpoint=False, dtype=dtype)

    if real:
        k = np.array([n*dk for n in range(M+1)], dtype=dtype)
    else:
        k = np.array([n*dk for n in range(0, M)] + [n*dk for n in range(M-N, 0)], dtype=dtype)

    return x, k, dx, dk



