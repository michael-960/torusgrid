from __future__ import annotations
from functools import partial
import pytest
from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
from torusgrid.fields import ComplexField1D, RealField1D
from scipy.fft import fft, ifft, rfft, irfft
import numpy as np

import torusgrid.misc.testutils as T

_cfg = T.get_config()
config = _cfg['fields']['1d']
gconfig = _cfg['global']
ARRAY_GEN_CFG = dict(
    min_offset=config['min_offset'], max_offset=config['max_offset'],
    min_scale=config['min_scale'], max_scale=config['max_scale']
)

def get_fft_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['fft_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['fft_tol']))

def get_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['floating_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['floating_tol']))


isreal = ['real', 'complex']

shapes = T.gen_shapes(config['num_samples'], ranks=[1], 
                      min_numel=config['min_numel'],
                      max_numel=config['max_numel'])

sizes = [T.gen_random_sequence(1, config['min_vol'], config['max_vol']) for _ in shapes]


ns = [shape[0] for shape in shapes]
ls = [size[0] for size in sizes]

reals = [['real', 'complex'][np.random.randint(0,2)] for _ in shapes]

precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]

argvals = dict(l=ls, n=ns, real_=reals, precision=precisions)


class TestField1D:
    """
    Test 1D complex and real fields functionalites
    """
    @T.parametrize('l', 'n', 'real_', 'precision', argvals=argvals)
    def test_props(self, l, n: int, real_: str, precision: PrecisionStr):

        if T.will_overflow((l,), (n,), precision):
            pytest.skip(msg='float overflow')

        real = real_ == 'real'
        dtype = T.real_dtype(precision)

        l = dtype(l)

        if not real:
            g = ComplexField1D(l, n, precision=precision)
            nk = n
        else:
            if n % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealField1D(l, n, precision=precision)
                    assert g.n == n+1
                    n += 1
            else:
                g = RealField1D(l, n, precision=precision)
            nk = n//2 + 1

        tol = get_tol(g.precision)
        dtype = T.real_dtype(precision)

        assert g.n == n
        assert g.shape == (n,)
        assert g.rank == 1
        assert g.last_fft_axis == 0
        assert g.fft_axes == (0,)
        assert g.precision is FloatingPointPrecision.cast(precision)
        assert g.shape_k == (nk,)

        assert g._fft is None
        assert g._ifft is None

        assert g.l == l
        assert len(g.size) == 1
        assert T.adaptive_allclose(g.size, l, **tol) # type: ignore

        assert T.adaptive_allclose(g.r, np.linspace(0, l, n+1, dtype=dtype)[:-1], **tol)
        dk = np.pi*2 / dtype(l)

        if real:
            k = np.array([dk*i for i in range(nk)])
        else:
            k = np.array([dk*i for i in range(nk//2)] + [dk*i for i in range(-(n-nk//2), 0)])

        assert T.adaptive_allclose(g.k, k, **tol)
        assert T.adaptive_allclose(g.k2, k**2, **tol)
        assert T.adaptive_allclose(g.k4, k**4, **tol)


    @T.parametrize('l', 'n', 'real_', 'precision', argvals=argvals)
    def test_copy(self, l: float, n: int, real_: bool, precision: PrecisionStr):

        if T.will_overflow((l,), (n,), precision):
            pytest.skip(msg='float overflow')

        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n, precision=precision)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n, precision=precision)

        tol = get_tol(g.precision)

        g.psi[...] = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real), precision=precision)

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.n == h.n
        assert g.l == h.l

        assert T.same_meta(g, h, **tol)


    @T.parametrize('l', 'n', 'real_', 'precision', argvals=argvals)
    def test_set_size(self, l: float, n: int, real_: bool, precision: PrecisionStr):

        if T.will_overflow((l,), (n,), precision):
            pytest.skip(msg='float overflow')

        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n, precision=precision)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n, precision=precision)

        tol = get_tol(g.precision)
        dtype = T.real_dtype(precision)

        for _ in range(config['num_setsize_trials']):
            l_ = T.gen_random_sequence(1, min_prod=config['min_vol'], max_prod=config['max_vol'])[0]
            l_ = dtype(l_)

            if T.will_overflow((l_,), (n,), precision): continue

            if np.random.rand() < 0.5:
                g.set_size(l_)
            else:
                g.set_size((l_,))

            assert T.adaptive_allclose(g.size, l_, **tol) # type: ignore
            assert T.adaptive_allclose(g.r, np.linspace(0, l_, n+1)[:-1], **tol)

            dk = np.pi*2 / dtype(l_)

            if real:
                k = np.array([dk*i for i in range(n//2+1)])
            else:
                k = np.array([dk*i for i in range(n//2)] + [dk*i for i in range(-(n-n//2), 0)])

            assert T.adaptive_allclose(g.k, k, **tol)
            assert T.adaptive_allclose(g.k2, k**2, **tol)
            assert T.adaptive_allclose(g.k4, k**4, **tol)


    @pytest.mark.slow
    @T.parametrize('l', 'n', 'real_', 'precision', argvals=argvals)
    def test_fft(self, l: float, n: int, real_: bool, precision: PrecisionStr):

        if T.will_overflow((l,), (n,), precision):
            pytest.skip(msg='float overflow')

        real = real_ == 'real'
        if not real:
            g = ComplexField1D(l, n, precision=precision)
        else:
            if n%2 == 1: n += 1
            g = RealField1D(l, n, precision=precision)

        tol = get_fft_tol(g.precision)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real), precision=precision)

            g.psi[...] = f
            assert np.array_equal(g.psi, f)
            g.fft() 
            fk = fft(f)
            
            if real:
                fk_r = rfft(f)
                assert T.adaptive_allclose(g.psi_k, fk_r, **tol) # type: ignore
            else:
                assert T.adaptive_allclose(g.psi_k, fk, **tol) # type: ignore

            g.ifft()
            f = ifft(fk)

            if real:
                f_r = irfft(fk_r) # type: ignore
                assert T.adaptive_allclose(g.psi, f_r, **tol) # type: ignore

            assert T.adaptive_allclose(g.psi, f, **tol) # type: ignore

            """
            Finally, to make sure that T.adaptive_atol_allclose is working as
            intended
            """
            assert not T.adaptive_allclose(
                    g.psi, 
                    T.gen_array(g.shape, **ARRAY_GEN_CFG, precision=precision),
                    **tol)

