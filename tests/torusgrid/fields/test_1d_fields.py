from __future__ import annotations
import pytest
from torusgrid.fields import ComplexField1D, RealField1D
from scipy.fft import fft, ifft, rfft, irfft
import numpy as np

import torusgrid.misc.testutils as T


NUM_SHAPES = 200
NUM_FFT_OPERATIONS = 128
NUM_FFT_THREADS = 8

NUM_SETSIZE_TRIALS = 8

FFT_RTOL = 1e-10
FFT_ATOL_FACTOR = 1e-8

MAX_OFFSET = 0
MIN_OFFSET = -1

MAX_SCALE = 1e9
MIN_SCALE = 1e-9

MIN_NUMEL = 1
MAX_NUMEL = 1e5

MIN_SIZE = 1e-8
MAX_SIZE = 1e8


TOL = dict(rtol=FFT_RTOL, atol_factor=FFT_ATOL_FACTOR)
ARRAY_GEN_CFG = dict(
    min_offset=MIN_OFFSET, max_offset=MAX_OFFSET,
    min_scale=MIN_SCALE, max_scale=MAX_SCALE
)


isreal = ['real', 'complex']

shapes = T.gen_shapes(NUM_SHAPES, ranks=[1], 
                      min_log_numel=np.log(MIN_NUMEL),
                      max_log_numel=np.log(MAX_NUMEL))

sizes = [T.gen_random_sequence(1, MIN_SIZE, MAX_SIZE) for _ in shapes]


ns = [shape[0] for shape in shapes]
ls = [size[0] for size in sizes]

_test_ids = [f'l={l:.4f} n={n}'
             for l,n in zip(ls,ns)]

def parametrize(f):
    return pytest.mark.parametrize(['l', 'n'], zip(ls, ns), ids=_test_ids)(f)


class TestField1D:
    """
    Test 1D complex and real fields functionalites
    """
    @pytest.mark.parametrize('real_', isreal)
    @parametrize
    def test_props(self, l: float, n: int, real_: str):
        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n)
            nk = n
        else:
            if n % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealField1D(l, n)
                    assert g.n == n+1
                    n += 1
            else:
                g = RealField1D(l, n)
            nk = n//2 + 1

        assert g.n == n
        assert g.shape == (n,)
        assert g.rank == 1
        assert g.last_fft_axis == 0
        assert g.fft_axes == (0,)
        assert g.precision == 'DOUBLE'
        assert g.shape_k == (nk,)

        assert g._fft is None
        assert g._ifft is None

        assert g.l == l
        assert len(g.size) == 1
        assert T.adaptive_atol_allclose(g.size, l, **TOL) # type: ignore

        assert T.adaptive_atol_allclose(g.r, np.linspace(0, l, n+1)[:-1], **TOL)
        dk = np.pi*2 / l

        if real:
            k = np.array([dk*i for i in range(nk)])
        else:
            k = np.array([dk*i for i in range(nk//2)] + [dk*i for i in range(-(n-nk//2), 0)])

        assert T.adaptive_atol_allclose(g.k, k, **TOL)
        assert T.adaptive_atol_allclose(g.k2, k**2, **TOL)
        assert T.adaptive_atol_allclose(g.k4, k**4, **TOL)
        assert T.adaptive_atol_allclose(g.k6, k**6, **TOL)


    @pytest.mark.parametrize('real_', isreal)
    @parametrize
    def test_copy(self, l: float, n: int, real_: bool):

        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n)

        g.psi[...] = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real))

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.shape == h.shape
        assert g.rank == h.rank
        assert g.n == h.n

        assert g.l == h.l

        assert g.shape_k == h.shape_k
        assert g.last_fft_axis == h.last_fft_axis
        assert g.fft_axes == h.fft_axes
        assert g.numel == h.numel
        assert g.isreal == h.isreal
        assert g.precision == h.precision

        assert np.array_equal(g.r, h.r)
        assert np.array_equal(g.k, h.k)
        assert np.array_equal(g.k2, h.k2)
        assert np.array_equal(g.k4, h.k4)
        assert np.array_equal(g.k6, h.k6)

        assert np.array_equal(g.size, h.size)
        assert g.volume == h.volume
        assert np.array_equal(g.dr, h.dr)
        assert np.array_equal(g.dk, h.dk)

        assert g.dv == h.dv


    @pytest.mark.parametrize('real_', isreal)
    @parametrize
    def test_set_size(self, l: float, n: int, real_: bool):
        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n)

        for _ in range(NUM_SETSIZE_TRIALS):
            l_ = T.gen_random_sequence(1, min_prod=MIN_SIZE, max_prod=MAX_SIZE)[0]

            if np.random.rand() < 0.5:
                g.set_size(l_)
            else:
                g.set_size((l_,))

            assert T.adaptive_atol_allclose(g.size, l_, **TOL) # type: ignore
            assert T.adaptive_atol_allclose(g.r, np.linspace(0, l_, n+1)[:-1], **TOL)

            dk = np.pi*2 / l_

            if real:
                k = np.array([dk*i for i in range(n//2+1)])
            else:
                k = np.array([dk*i for i in range(n//2)] + [dk*i for i in range(-(n-n//2), 0)])

            assert T.adaptive_atol_allclose(g.k, k, **TOL)
            assert T.adaptive_atol_allclose(g.k2, k**2, **TOL)
            assert T.adaptive_atol_allclose(g.k4, k**4, **TOL)
            assert T.adaptive_atol_allclose(g.k6, k**6, **TOL)


    @pytest.mark.slow
    @pytest.mark.parametrize('real_', isreal)
    @parametrize
    def test_fft(self, l: float, n: int, real_: bool):
        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real))

            g.psi[...] = f
            assert np.array_equal(g.psi, f)
            g.fft() 
            fk = fft(f)
            
            if real:
                fk_r = rfft(f)
                assert T.adaptive_atol_allclose(g.psi_k, fk_r, **TOL) # type: ignore
            else:
                assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore

            g.ifft()
            f = ifft(fk)

            if real:
                f_r = irfft(fk_r) # type: ignore
                assert T.adaptive_atol_allclose(g.psi, f_r, **TOL) # type: ignore

            assert T.adaptive_atol_allclose(g.psi, f, **TOL) # type: ignore

            """
            Finally, to make sure that T.adaptive_atol_allclose is working as
            intended
            """
            assert not T.adaptive_atol_allclose(
                    g.psi, 
                    T.gen_array(g.shape, MIN_OFFSET, MAX_OFFSET, MIN_SCALE, MAX_SCALE),
                    **TOL)

