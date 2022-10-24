from typing import List, Sequence, Tuple

import pytest
import numpy as np
from torusgrid.fields import ComplexField

from scipy.fft import fftn, ifftn

def prod(arr: Sequence[float]):
    """product of all the elements in an array"""
    r = 1
    for a in arr: r *= a
    return r


def get_shape(rank: int, min_log_numel: float, max_log_numel: float):
    """Generate shape"""
    log_numel = np.random.rand() * (max_log_numel - min_log_numel) + min_log_numel

    seps = np.random.rand(rank-1).tolist()
    seps = [0] + sorted(seps) + [1]
    
    shape_ = []

    for i in range(rank):
        s = np.exp(log_numel * (seps[i+1] - seps[i]))
        shape_.append(int(s))

    return tuple(shape_)


def get_shapes(n: int, ranks: Sequence[int],
               max_log_numel: float, min_log_numel: float) -> List[Tuple[int,...]]:
    """Generate multiple shapes"""
    res = []
    for _ in range(n):
        rank = np.random.choice(ranks)
        shape = get_shape(rank, max_log_numel=max_log_numel, min_log_numel=min_log_numel)
        res.append(shape)
    return res

def get_size(shape: Tuple[int,...], min, max):
    """
    Return a random size based on shape
    """
    rank = len(shape)
    size = []
    for _ in range(rank):
        size.append(np.random.rand()*(max-min) + min)

    return tuple(size)


def get_fft_axes(rank: int):
    """Generate FFT axes"""
    n_ax = np.random.randint(1, rank+1)
    return tuple(np.random.choice(range(rank), replace=False, size=n_ax).tolist())


def get_test_id(size, shape, fft_axes):
    """Test IDs"""
    size_str = '(' + ','.join([f'{s:.3f}' for s in size]) + ')'
    shape_str = '(' + ','.join([str(s) for s in shape]) + ')'
    fft_ax_str = '(' + ','.join([str(s) for s in fft_axes]) + ')'

    return f'size={size_str} shape={shape_str} fft={fft_ax_str}'


shapes = get_shapes(100, ranks=[1]*2 + [2]*5 + [3]*3 + [4,5,6,7,8], min_log_numel=4, max_log_numel=np.log(2e5))
sizes = [get_size(shape, min=0.1, max=1e3) for shape in shapes]
fft_axeses = [get_fft_axes(len(shape)) for shape in shapes]
_test_ids = [get_test_id(size,shape,fft_ax) for size,shape,fft_ax in zip(sizes, shapes, fft_axeses)]


def parametrize(f):
   return pytest.mark.parametrize(['size', 'shape', 'fft_axes'], zip(sizes, shapes, fft_axeses), ids=_test_ids)(f)

@parametrize
def test_props(size, shape, fft_axes): 
    """
    Test field properties
    """
    g = ComplexField(size=size, shape=shape, fft_axes=fft_axes)
    assert g.shape == shape
    assert g.psi.shape == shape
    assert g.rank == len(shape)
    assert np.allclose(g.size, size)
    assert g.numel == prod(shape)

    assert g.r.shape[0] == len(shape)
    assert g.r.shape[1:] == shape

    assert np.size(g.r) == g.rank * prod(shape)
    
    r = []
    for l, n in zip(size, shape):
        r.append(np.linspace(0, l, n+1)[:-1])

    r = np.stack(np.meshgrid(*r, indexing='ij'), axis=0)
    assert np.allclose(r, g.r, rtol=1e-13, atol=1e-13)

    k = []
    for L, N in zip(size, shape):
        dk = np.pi*2 / L
        M = N // 2
        _k = [i*dk for i in range(M)] + [i*dk for i in range(-(N-M), 0)]
        k.append(np.array(_k))

    k = np.stack(np.meshgrid(*k, indexing='ij'), axis=0)

    assert np.allclose(k, g.k, rtol=1e-13, atol=1e-13)
    k2 = np.sum(k**2, axis=0)
    assert np.allclose(k2, g.k2, rtol=1e-13, atol=1e-13)
    assert np.allclose(k2**2, g.k4, rtol=1e-13, atol=1e-13)
    assert np.allclose(k2**3, g.k6, rtol=1e-13, atol=1e-13)


    assert g._fft is None
    assert g._ifft is None

    g.initialize_fft()
    
    assert g._fft is not None
    assert g._ifft is not None

    assert not g.isreal


@parametrize
def test_fft(size, shape, fft_axes):
    """Test FFT"""
    g = ComplexField(size=size, shape=shape, fft_axes=fft_axes)

    with pytest.raises(TypeError):
        g.fft()

    g.initialize_fft(threads=4)

    N = 10
    for _ in range(N):
        a = np.random.rand(*shape)
        b = np.random.rand(*shape)
        f = a + 1j*b

        g.psi[...] = f

        assert np.all(g.psi[...] == f)

        g.fft()

        fk = fftn(f, axes=fft_axes)
        assert np.allclose(g.psi_k, fk, rtol=1e-9, atol=1e-9) # type: ignore

        f_ = ifftn(fk, axes=fft_axes)
        assert np.allclose(g.psi, f_, rtol=1e-9, atol=1e-9) # type: ignore


@parametrize
def test_copy(size, shape, fft_axes):
    g = ComplexField(size=size, shape=shape, fft_axes=fft_axes)

    a = np.random.rand(*shape)
    b = np.random.rand(*shape)
    f = a + 1j*b

    g.psi[...] = f

    h = g.copy()

    assert np.all(h.size == g.size)
    assert h.numel == g.numel
    assert np.all(h.psi == g.psi)

    assert h.psi is not g.psi
    assert h.psi_k is not g.psi_k

    g.initialize_fft()

    g.fft()

    assert not np.all(g.psi_k == h.psi_k)
    
    h.initialize_fft()
    h.fft()

    assert np.all(g.psi_k == h.psi_k)


real_dtype_table = {
    'single': np.single,
    'double': np.double,
    'longdouble': np.longdouble
}

complex_dtype_table = {
    'single': np.csingle,
    'double': np.cdouble,
    'longdouble': np.clongdouble
}

nbytes_table = {'single': 4, 'double': 8, 'longdouble': 16}


@parametrize
@pytest.mark.parametrize('precision', ['single', 'double', 'longdouble'])
def test_precision(size, shape, fft_axes, precision):
    g = ComplexField(size=size, shape=shape, fft_axes=fft_axes, precision=precision)

    assert g.psi.dtype == complex_dtype_table[precision]
    assert g.psi_k.dtype == complex_dtype_table[precision]

    assert g.psi.nbytes == g.numel * nbytes_table[precision] * 2
    assert g.psi_k.nbytes == g.numel * nbytes_table[precision] * 2
    
    assert g.size.dtype == real_dtype_table[precision]
    assert g.r.dtype == real_dtype_table[precision]
    assert g.k.dtype == real_dtype_table[precision]

    assert g.k2.dtype == real_dtype_table[precision]
    assert g.k4.dtype == real_dtype_table[precision]
    assert g.k6.dtype == real_dtype_table[precision]

