from __future__ import annotations
import numpy as np
import numpy.typing as npt

from ...core import FloatLike

from ...grids import Grid
from ...fields import Field


def adaptive_allclose(
        f1: npt.ArrayLike, f2: npt.ArrayLike, *,
        rtol1: FloatLike= 0, rtol2: FloatLike= 0):
    """
    Compare two arrays symmetrically

    Two arrays are close if
        abs(a-b) <= rtol1 * (magnitude) + rtol2 * (deviation)

    """
    f1_ = np.asarray(f1)
    f2_ = np.asarray(f2)

    dev1 = np.sqrt(np.mean(np.abs(f1_-np.mean(f1_))**2))
    dev2 = np.sqrt(np.mean(np.abs(f1_-np.mean(f1_))**2))

    dev = np.sqrt((dev1**2+dev2**2)/2)

    mag = np.sqrt((np.abs(f1_**2) + np.abs(f2_**2)) / 2)

    return np.all(distance(f1, f2) <= rtol1 * mag + rtol2 * dev) # type: ignore


def distance(f1, f2) -> np.ndarray:
    f1_ = np.asarray(f1)
    f2_ = np.asarray(f2)
    return np.abs(f1_ - f2_)





def same_dimensions(g: Grid, h: Grid, *, rtol1=0, rtol2=0):
    r = all([
        g.shape == h.shape,
        g.rank == h.rank,
        g.numel == h.numel,
        g.shape_k == h.shape_k,
    ])

    if isinstance(g, Field): 
        assert isinstance(h, Field)
        r = r and all([
            isinstance(h, Field),
            adaptive_allclose(g.size, h.size, rtol1=rtol1, rtol2=rtol2),
            adaptive_allclose(g.r, h.r, rtol1=rtol1, rtol2=rtol2),
            adaptive_allclose(g.k, h.k, rtol1=rtol1, rtol2=rtol2),
            adaptive_allclose(g.volume, h.volume, rtol1=rtol1, rtol2=rtol2),
            adaptive_allclose(g.dv, h.dv, rtol1=rtol1, rtol2=rtol2)
        ])

    return r


def same_meta_except_dimensions(g: Grid, h: Grid):
    return all([
        type(g) is type(h),
        g.fft_axes == h.fft_axes,
        g.precision is h.precision
    ])


def same_meta(g: Grid, h: Grid, *, rtol1=0, rtol2=0):

    r1 = same_dimensions(g, h, rtol1=rtol1, rtol2=rtol2)
    r2 = same_meta_except_dimensions(g, h)
    return r1 and r2


