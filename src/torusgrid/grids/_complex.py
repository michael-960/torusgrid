from __future__ import annotations

from typing import Tuple, final

import numpy as np
import pyfftw


from ..core import get_complex_dtype

from ._base import Grid


class ComplexGrid(Grid[np.complexfloating]):
    """
    A ComplexGridND object is a complex array of shape (d1, d2, .., dN)
    equipped with fourier transform. No length scales are associated with the
    grid. 
    """
    @final
    def _init_grid_data(self, shape: Tuple[int, ...]):
        self._psi = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))
        self._psi_k = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))

    @property
    def isreal(self): return False

    # def set_psi(self, 
    #         psi1: Union[complex,
    #                     npt.NDArray[np.complexfloating],
    #                     npt.NDArray[np.floating]]
    #     ) -> None:
    #     """
    #     Set grid data.
    #
    #     Parameters: psi1: new grid data, can be either scalar (float) or
    #     np.ndarray. If a scalar is given, all entries are set to the given value.
    #     """
    #     if not np.isscalar(psi1):
    #         assert isinstance(psi1, np.ndarray)
    #         if psi1.shape != self.shape:
    #             raise ValueError(f'array has incompatible shape {psi1.shape} with {self.shape}')
    #
    #     self.psi[...] = psi1

    # def __init__(self, 
    #         shape: Tuple[int, ...], *,
    #         precision: PrecisionStr='double',
    #         fft_axes: Optional[Tuple[int,...]]=None
    #     ):
    #     """
    #     Set the resolution (i.e. shape).
    #     
    #     Parameters: shape: tuple of integers (d1, d2, ..., dN),
    #     """
    #     self._precision = precision
    #
    #     self._psi = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))
    #     self._psi_k = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))
    #
    #     if fft_axes is None:
    #         self._fft_axes = tuple(np.arange(self.rank))
    #     else:
    #         self._fft_axes = fft_axes


    # def save(self, fname: str, verbose=False) -> None:
    #     """Save the grid data into a file.
    #
    #     Parameters:
    #         fname: the base file name. A .grid extension will be appended.
    #         verbose: whether to print out details
    #     """
    #     tmp_name = f'{fname}.tmp.file'
    #     if verbose:
    #         self.yell(f'dumping profile data to {fname}.grid')
    #     np.savez(tmp_name, **self.export_state()) 
    #     shutil.move(f'{tmp_name}.npz', f'{fname}.grid')

    # def export_state(self) -> dict:
    #     """Export the grid state to a dictionary
    #     """
    #     state = {'psi': self.psi.copy()}
    #     return state
