from typing import Type, TypeVar
import numpy as np

from ..grids import Grid
from ..core import FloatingPointPrecision

import numpy.typing as npt


T = TypeVar('T', bound=Grid)



def save(obj: Grid, path: str):
    """
    Save grid or field object to file
    """
    metadata = obj.metadata()
    with open(path, 'wb') as f:
        np.savez(f, **metadata, psi=obj.psi)


def load(cls: Type[T], path: str, *,
         autometa: bool = False) -> T:
    """
    Load a grid or field from file
    Parameters:
        cls: the class
        path: file path

        autometa: 
            Whether to automatically supply metadata if metadata is absent from
            file. Specifically, 'shape' and precision' will be deduced from
            array's shape and dtype, while 'fft_axes' will be set to all axes.
    """

    with open(path, 'rb') as f:
        d = np.load(f)
        psi = d['psi']
        metadata = {k:v for k,v in d.items() if k != 'psi'}

    if autometa:
        if 'precision' not in metadata.keys():
            metadata['precision'] = FloatingPointPrecision.from_dtype(psi.dtype).name

        if 'fft_axes' not in metadata.keys():
            metadata['fft_axes'] = tuple(i for i in range(len(psi.shape)))

        if 'shape' not in metadata.keys():
            metadata['shape'] = psi.shape

    return cls.from_array(psi, metadata=metadata)


