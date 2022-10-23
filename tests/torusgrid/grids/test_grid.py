from typing import List, Tuple
from torusgrid import grids
import pytest

from scipy.fft import fftn, ifftn
import numpy as np

import time




shapes: List[Tuple[int,...]] = [
    (3,), (4,), (14,), (16,), (64,), (100,), (128,), (256,), (1024,), (2048,),
    (3,3), (3,4), (3,16), (4,16), (16,16), (16,32), (32,16), (32,128),
    (128,128), (16,1024), (16,1000), (16,1025), (16,2048), (256,256), (256,257), (512,512),
    (16,8192), (32,8192), (60,8000), (60,8192), (64,8192), (16, 16384), (32, 16384),
    (33, 16383), (33, 16385), (30, 10000), (1,10240),
    (4,4,5), (4,4,16), (4,8,16), (4,32,512), (32, 412, 9), (9,8,7), (32,32,32), (64,64,64),
    (128,128,3), (128,128,4), (128,128,32), (512, 16, 8), (64,64,64),
    (4,5,6,7), (16,17,18,19), (16,16,16,16), (32,32,32,32), (64,64,64,2),
    (7,8,9,16,16), (9, 9, 9, 9, 9)
]

@pytest.mark.parametrize('shape', shapes)
def test_props(shape: Tuple[int,...]):
    """
    Test grid properties
    """
    rank = len(shape)
    n_fft_axes = np.random.randint(0, rank+1)
    fft_axes = tuple(
            np.random.choice(range(rank), size=n_fft_axes, replace=False).tolist()
            )

    scale = np.random.rand()*9.9 + 0.1

    g = grids.ComplexGrid(shape, fft_axes=fft_axes)

    f = (2*np.random.rand(*shape) - 1) * scale # type: ignore

    g.psi[...] = f

    assert np.all(g.psi == f)
    assert g.rank == rank
    assert g.rank == len(shape)
    assert g.numel == np.prod(shape)
    assert g.numel == np.size(f)


@pytest.mark.parametrize('shape', shapes)
def test_fft(shape):
    """
    Test whether fft() and ifft() yield the same results as scipy's fftn() and
    ifftn()
    """
    rank = len(shape)
    n_fft_axes = np.random.randint(0, rank+1)
    N = 20

    fft_axes = tuple(
            np.random.choice(range(rank), size=n_fft_axes, replace=False).tolist()
            )

    g = grids.ComplexGrid(shape, fft_axes=fft_axes)
    
    with pytest.raises(RuntimeError):
        g.fft()

    g.initialize_fft()

    for _ in range(N):
        scale = np.random.rand()*99.9 + 0.1
        f = (2*np.random.rand(*shape) - 1) * scale # type: ignore
        g.psi[...] = f

        assert np.allclose(g.psi, f, rtol=1e-7, atol=1e-7)

        g.fft()
        fk: np.ndarray = fftn(f, axes=fft_axes) # type: ignore
        assert np.allclose(g.psi_k, fk, rtol=1e-7, atol=1e-7)

        g.ifft()
        f: np.ndarray = ifftn(fk, axes=fft_axes) # type: ignore
        assert np.allclose(g.psi, f, rtol=1e-7, atol=1e-7)

    g.psi[...] = 0
    
    g.fft()
    assert np.allclose(g.psi_k, 0)

    
@pytest.mark.parametrize('shape', shapes)
def test_copy(shape):
    """
    Test the copy() function
    """
    rank = len(shape)
    n_fft_axes = np.random.randint(0, rank+1)

    fft_axes = tuple(
            np.random.choice(range(rank), size=n_fft_axes, replace=False).tolist()
            )

    g = grids.ComplexGrid(shape, fft_axes=fft_axes)

    g.psi[...] = np.random.rand(*shape)
    h = g.copy()

    assert np.all(g.psi == h.psi)
    assert g.psi is not h.psi
    assert g.psi_k is not h.psi_k

    assert g.rank == h.rank
    assert g.shape == h.shape

    g.initialize_fft()
    h.initialize_fft()

    g.fft()
    h.fft()

    assert np.all(g.psi_k == h.psi_k)

    g.ifft()
    h.ifft()

    assert np.all(g.psi == h.psi)



@pytest.mark.parametrize('shape', shapes)
def test_time(shape):
    """
    Assert scipy's fft is slower
    """
    rank = len(shape)
    n_fft_axes = np.random.randint(0, rank+1)
    fft_axes = tuple(
            np.random.choice(range(rank), size=n_fft_axes, replace=False).tolist()
            )
    scale = np.random.rand()*9.9 + 0.1

    g = grids.ComplexGrid(shape, fft_axes=fft_axes)


    g.psi[...] = f

    g.initialize_fft()

    N = 100


    start = time.time()
    for _ in range(N):
        fk = fftn(f, axes=fft_axes)

    end = time.time()
    
    time_scipy = end - start


    start = time.time()
    for _ in range(N):
        g.fft()
        # fk = g.psi_k

    end = time.time()
    time_tg = end - start

    assert time_tg < time_scipy











