from __future__ import annotations
import matplotlib.pyplot as plt

from ..core import FloatLike
from ..fields import RealField2D


def plot_2d(
    ax: plt.Axes, field: RealField2D, *,
    cmap: str='jet', colorbar: bool=True,
    vmin: FloatLike|None=None,
    vmax: FloatLike|None=None,
):
    """
    Plot 2d real field
    """
    mesh = ax.pcolormesh(field.x, field.y, field.psi,
                         cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(mesh, ax, orientation='horizontal')

    return mesh
