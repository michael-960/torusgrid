from __future__ import annotations
from typing import Sequence, Tuple, Optional
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from michael960lib.math import fourier

from ..typing import PrecisionStr, SizeLike

from ..grids import ComplexGrid

from ._base import Field



class ComplexField(ComplexGrid, Field[np.complexfloating]):
    '''
    A field is a grid with length scales. While the grid shape is fixed for a
    particular instance, the system size (lengths in each dimension) can be
    changed with set_size().

    '''
    def __init__(self,
            size: Sequence[np.floating|float] | npt.NDArray[np.floating],
            shape: Tuple[int, ...], *,
            precision: PrecisionStr = 'double',
            fft_axes: Optional[Tuple[int,...]] = None
        ):
        super().__init__(
            shape,
            precision=precision, 
            fft_axes=fft_axes
        )
        self._init_coordinate_vars()
        self.set_size(size)

    def _update_coordinate_vars(self) -> None:
        
        R, K, DR, DK = [], [], [], []
        for i in range(self.rank):
            r, k, dr, dk = fourier.generate_xk(self.size[i], self.shape[i]) 
            R.append(r)
            K.append(k)
            DR.append(dr)
            DK.append(dk)

        self._R[...] = np.meshgrid(*R, indexing='ij')
        self._K[...] = np.meshgrid(*K, indexing='ij')
        self._dR[...] = np.array(DR)
        self._dK[...] = np.array(DK)


    def copy(self) -> Self:
        field1 = self.__class__(
                self.size, self.shape,
                precision=self._precision,
                fft_axes=self._fft_axes
        )

        field1.set_psi(self.psi)
        return field1

    # def save(self, fname: str, verbose=False):
    #     tmp_name = f'{fname}.tmp.file'
    #     if verbose:
    #         self.yell(f'dumping field data to {fname}.field')
    #     np.savez(tmp_name, **self.export_state()) 
    #     shutil.move(f'{tmp_name}.npz', f'{fname}.field')





