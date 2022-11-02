from __future__ import annotations
from typing import Any, Optional, cast, overload
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt

import numpy as np

from ..core import FloatLike
from ..fields import RealField2D



@overload
def plot_2d(
    field: RealField2D, *, 
    colorbar: bool=True, cmap: str='jet', 
    vmin: FloatLike|None=None,
    vmax: FloatLike|None=None
) -> None: ...


@overload
def plot_2d(
    field: RealField2D, *, ax: plt.Axes,
    colorbar: bool=True, cmap: str='jet', 
    vmin: FloatLike|None=None,
    vmax: FloatLike|None=None
) -> QuadMesh: ...


def plot_2d(
    field: RealField2D, *,
    ax: Optional[plt.Axes]=None,
    colorbar: bool = True,
    cmap: str='jet',
    vmin: FloatLike|None=None,
    vmax: FloatLike|None=None
):
    """
    Plot 2d real field.
    
    If ax is specified, then a QuadMesh object is returned 
    and plt.show() will not be invoked.

    """
    _ax: plt.Axes
    if ax is None:
        _ax = cast(plt.Axes, plt.subplot())
    else:
        _ax = ax

    mesh = _ax.pcolormesh(field.x.astype(np.double), 
                          field.y.astype(np.double),
                          field.psi.astype(np.double),
                         cmap=cmap, vmin=vmin, vmax=vmax)
    _ax.set_aspect('equal')

    if colorbar:
        plt.colorbar(mesh, ax=_ax)

    if ax is not None:
        return mesh
    else:
        plt.show()


