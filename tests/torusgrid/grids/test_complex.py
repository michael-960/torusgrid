from typing import List, Tuple
from torusgrid import grids
from scipy.fft import fftn, ifftn
import numpy as np
import pytest
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
    n_fft_axes = np.random.randint(1, rank+1)
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
    
    with pytest.raises(TypeError):
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
    n_fft_axes = np.random.randint(1, rank+1)

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


@pytest.mark.parametrize(
        'shape_fft_axes', 
        [((4,), (0,)), 
         ((16,), (0,)),
         ((32,), (0,)), 
         ((64,), (0,)),
         ((128,), (0,)),
         ((512,), (0,)),
         ((2048,), (0,)),
         ((4096,), (0,)),
         ((8192,), (0,)),
         ((16,16,), (0,1)), ((16,16,), (1,)),
         ((32,16,), (0,1)), ((32,16,), (0,)),
         ((64,128,), (0,1)), ((64,128,), (0,)),
         ((32,256,), (0,1)), ((32,256,), (1,)),
         ((256,256,), (0,1)), ((256,256,), (1,)),
         ((512,8,), (1,0)), ((512,8,), (0,)),
         ((512,64,), (0,1)), ((512,64,), (1,)),
         ((1024,256,), (0,1)), ((1024,256,), (0,)),
         ((2048,128,), (0,1)), ((2048,128,), (1,)),
         ((8,8,32), (0,1,2)), ((8,8,32), (0,2)), ((8,8,32), (2,1)),
         ((32,64,16), (0,1,2)), ((32,64,16), (1,2)), ((32,64,16), (0,)),
         ((64,64,64), (0,2,1)), ((64,64,64), (2,0)), ((64,64,64), (2,)),
         ((256,128,64), (0,1,2)), ((256,128,64), (2,)), ((256,128,64), (2,1,0)), ((256,128,64), (1,0)),
])
def test_time(shape_fft_axes: Tuple[Tuple[int,...], Tuple[int,...]]):

    shape, fft_axes = shape_fft_axes
    g = grids.ComplexGrid(shape, fft_axes=fft_axes)

    if g.numel > 1e4:
        g.initialize_fft(threads=4)
    elif g.numel > 1e3:
        g.initialize_fft(threads=2)
    else:
        g.initialize_fft()

    scale = np.random.rand()*9.9 + 0.1
    a: np.ndarray = (2*np.random.rand(*shape) - 1) * scale # type: ignore
    b: np.ndarray = (2*np.random.rand(*shape) - 1) * scale # type: ignore
    f = a + 1j*b

    g.psi[...] = f

    N = 512

    start = time.time()
    for _ in range(N):
        fftn(f, axes=fft_axes)

    end = time.time()

    t_scipy = end - start


    start = time.time()
    for _ in range(N):
        g.fft()
    end = time.time()

    t_tg = end - start
    
    assert t_tg < t_scipy

