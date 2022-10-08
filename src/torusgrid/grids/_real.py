from __future__ import annotations
from typing import Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from pyfftw import pyfftw
import warnings


from ..typing import get_complex_dtype, get_real_dtype, PrecisionStr

from ._base import Grid


class RealGridND(Grid):
    '''
    A RealGridND object is a real array of shape (d1, d2, .., dN) equipped
    with fourier transform. No length scales are associated with the grid.

    The generic type variables are the dtypes of self.psi & self.psi_k.
    
    '''

    def __init__(self,
            shape: Tuple[int, ...], *,
            precision: PrecisionStr = 'double',
            fft_axes: Optional[Tuple[int,...]] = None
        ):
        '''Set the resolution (i.e. shape).
        
        Parameters: shape: tuple of integers (d1, d2, ..., dN),

        The last axis in self.fft_axes must be even
        (if not it will be made even automatically)

        '''


        self.psi: npt.NDArray[np.floating]
        self.psi_k: npt.NDArray[np.complexfloating]

        self._isreal = True
        self._precision: PrecisionStr = precision

        if fft_axes is None:
            self._fft_axes = tuple(np.arange(len(shape)))
        else:
            self._fft_axes = fft_axes

        last_axis = self._fft_axes[-1]

        if shape[last_axis] % 2 != 0:
            warnings.warn('odd resolution on the last axis will be automatically made even for RFFT')
            shape_list = list(shape)
            shape_list[last_axis] += 1
            shape = tuple(shape_list)

        shape_k = list(shape)

        shape_k[last_axis] = shape_k[last_axis]//2 + 1

        shape_k = tuple(shape_k)

        self.psi = pyfftw.zeros_aligned(
                shape, dtype=get_real_dtype(self._precision))

        self.psi_k = pyfftw.zeros_aligned(
                shape_k, dtype=get_complex_dtype(self._precision))

    def copy(self) -> Self:
        '''Generate a new object with the same grid data.
        '''
        grid1 = self.__class__(
                    self.shape,
                    precision=self._precision,
                    fft_axes=self._fft_axes
                )
        grid1.set_psi(self.psi)
        return grid1
    
    def set_psi(self, psi1: Union[float, npt.NDArray[np.floating]]):
        if not np.isscalar(psi1):
            assert isinstance(psi1, np.ndarray)
            if psi1.shape != self.shape:
                raise ValueError(f'array has incompatible shape {psi1.shape} with {self.shape}')

        if not np.all(np.isreal(psi1)):
            raise ValueError(f'array is complex') 
        self.psi[...] = psi1



