from __future__ import annotations
from typing_extensions import Self

from typing import Tuple, Union, Optional, final

import numpy as np
import numpy.typing as npt
import pyfftw

from ..typing import PrecisionStr, get_complex_dtype

from ._base import Grid


class ComplexGridND(Grid):
    '''
    A ComplexGridND object is a complex array of shape (d1, d2, .., dN)
    equipped with fourier transform. No length scales are associated with the
    grid. 
    '''
    def __init__(self, 
            shape: Tuple[int, ...], *,
            precision: PrecisionStr='double',
            fft_axes: Optional[Tuple[int,...]]=None
        ):

        self.psi: npt.NDArray[np.complexfloating]
        self.psi_k: npt.NDArray[np.complexfloating]

        self._isreal = False
        self._precision: PrecisionStr = precision

        self.psi = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))
        self.psi_k = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))

        if fft_axes is None:
            self._fft_axes = tuple(np.arange(self.rank))
        else:
            self._fft_axes = fft_axes

        
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

    def set_psi(self, 
            psi1: Union[complex, 
                        npt.NDArray[np.complexfloating],
                        npt.NDArray[np.floating]]
        ) -> None:
        '''Set grid data.

        Parameters: psi1: new grid data, can be either scalar (float) or
        np.ndarray. If a scalar is given, all entries are set to the given value.
        '''
        if not np.isscalar(psi1):
            assert isinstance(psi1, np.ndarray)
            if psi1.shape != self.shape:
                raise ValueError(f'array has incompatible shape {psi1.shape} with {self.shape}')
        self.psi[...] = psi1

    # def save(self, fname: str, verbose=False) -> None:
    #     '''Save the grid data into a file.
    #
    #     Parameters:
    #         fname: the base file name. A .grid extension will be appended.
    #         verbose: whether to print out details
    #     '''
    #     tmp_name = f'{fname}.tmp.file'
    #     if verbose:
    #         self.yell(f'dumping profile data to {fname}.grid')
    #     np.savez(tmp_name, **self.export_state()) 
    #     shutil.move(f'{tmp_name}.npz', f'{fname}.grid')

    # def export_state(self) -> dict:
    #     '''Export the grid state to a dictionary
    #     '''
    #     state = {'psi': self.psi.copy()}
    #     return state
