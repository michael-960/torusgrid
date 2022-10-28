from __future__ import annotations
from functools import partial
from typing import Tuple
import pytest
from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
from torusgrid.fields import ComplexField2D, RealField2D
from scipy.fft import fft2, ifft2,rfft2, irfft2
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

shapes = T.gen_shapes(config['num_samples'], ranks=[2], 
                      min_numel=config['min_numel'],
                      max_numel=config['max_numel'])

sizes = [T.gen_random_sequence(2, config['min_vol'], config['max_vol']) for _ in shapes]

fft_axeses = [[None,(0,1),(0,),(1,),(1,0)][np.random.randint(0,5)] for _ in shapes]

reals = [['real', 'complex'][np.random.randint(0,2)] for _ in shapes]

precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]

argvals = dict(size=sizes, shape=shapes, fft_axes=fft_axeses, real_=reals, precision=precisions)

class TestField2D:

    @T.parametrize('size', 'shape', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_props(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real_: str, precision: PrecisionStr
    ):
        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')

        dtype = T.real_dtype(precision)
        size = [dtype(s) for s in size]

        real = (real_ == 'real')

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]


        if not real:
            g = ComplexField2D(*size, *shape, fft_axes=fft_axes, precision=precision)
        else:
            if shape[last_fft_axis] % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealField2D(*size, *shape, fft_axes=fft_axes, precision=precision)
                assert g.shape[last_fft_axis] == shape[last_fft_axis] + 1
                shape = T.even_shape(shape, fft_axes)
            else:
                g = RealField2D(*size, *shape, fft_axes=fft_axes, precision=precision)

        tol = get_tol(g.precision)

        assert g.rank == 2
        assert last_fft_axis in [0,1]
        assert g._fft is None
        assert g._ifft is None

        assert g.last_fft_axis == fft_axes[-1]
           
        assert g.nx == shape[0]
        assert g.ny == shape[1]

        if not real:
            shape_k = list(shape)
        else:
            shape_k = list(shape)
            shape_k[last_fft_axis] //= 2
            shape_k[last_fft_axis] += 1

        assert g.psi_k.shape == tuple(shape_k)
        assert g.shape_k == tuple(shape_k)

        assert g.r.shape == (2, *g.shape)
        assert g.k.shape == (2, *g.shape_k)

        assert g.precision is FloatingPointPrecision.cast(precision)

        r = []
        k = []
        for ax in [0,1]:
            L, N = size[ax], shape[ax]
            r.append(np.linspace(0, L, N+1, dtype=dtype)[:-1])
            dk = np.pi*2 / dtype(L)

            if real and ax == fft_axes[-1]:
                assert N % 2 == 0
                k.append(np.array([dk*i for i in range(N//2+1)]))
            else:
                k.append(np.array(
                    [i*dk for i in range(N//2)] +
                    [i*dk for i in range(-(N-N//2), 0)]
                ))
        r = np.stack(np.meshgrid(*r, indexing='ij'))
        k = np.stack(np.meshgrid(*k, indexing='ij'))

        k2 = k[0]**2 + k[1]**2

        assert T.adaptive_allclose(g.r, r, **tol)
        assert T.adaptive_allclose(g.k, k, **tol)
        assert g.r.shape == r.shape
        assert g.k.shape == k.shape
        assert T.adaptive_allclose(g.k2, k2, **tol)
        assert T.adaptive_allclose(g.k4, k2**2, **tol)


    @T.parametrize('size', 'shape', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_copy(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real_: str, precision: PrecisionStr
    ):

        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')

        dtype = T.real_dtype(precision)

        real = (real_ == 'real')
        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes, precision=precision)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape = T.even_shape(shape, fft_axes)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes, precision=precision)

        tol = get_tol(g.precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real), precision=precision)

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert T.same_meta(g, h, **tol)

        assert g.nx == h.nx
        assert g.ny == h.ny
        assert g.lx == h.lx
        assert g.ly == h.ly


    @T.parametrize('size', 'shape', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_set_size(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real_: str, precision: PrecisionStr
    ):
        
        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')

        dtype = T.real_dtype(precision)

        real = (real_ == 'real')

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes, precision=precision)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes, precision=precision)

        tol = get_tol(g.precision)

        for _ in range(config['num_setsize_trials']):
            lx_, ly_ = T.gen_random_sequence(
                    2, min_prod=config['min_vol'], max_prod=config['max_vol'])

            lx_ = dtype(lx_)
            ly_ = dtype(ly_)

            if T.will_overflow((lx_,ly_), shape, precision): continue

            if np.random.rand() < 0.5:
                g.set_size(lx_, ly_)
            else:
                g.set_size((lx_, ly_))

            assert T.adaptive_allclose(g.size, (lx_, ly_), **tol) # type: ignore

            r = []
            k = []
            for ax, L, N in zip([0,1], (lx_, ly_), shape):
                r.append(np.linspace(0, L, N+1, dtype=dtype)[:-1])
                dk = np.pi*2 / L

                if real and ax == fft_axes[-1]:
                    assert N % 2 == 0
                    k.append(np.array([dk*i for i in range(N//2+1)]))
                else:
                    k.append(np.array(
                        [i*dk for i in range(N//2)] +
                        [i*dk for i in range(-(N-N//2), 0)]
                    ))
            r = np.stack(np.meshgrid(*r, indexing='ij'))
            k = np.stack(np.meshgrid(*k, indexing='ij'))

            k2 = k[0]**2 + k[1]**2

            assert T.adaptive_allclose(g.r, r, **tol)
            assert T.adaptive_allclose(g.k, k, **tol)
            assert T.adaptive_allclose(g.k2, k2, **tol)
            assert T.adaptive_allclose(g.k4, k2**2, **tol)


    @pytest.mark.slow
    @T.parametrize('size', 'shape', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_fft(
        self, 
        size: Tuple[float,float], shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real_: str, precision: PrecisionStr
    ):

        if T.will_overflow(size, shape, precision):
            pytest.skip()

        dtype = T.real_dtype(precision)
        size = [dtype(s) for s in size]

        real = (real_ == 'real')

        if fft_axes is None:
            fft_axes = (0,1)

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes, precision=precision)
        else:
            if shape[fft_axes[-1]] % 2 == 1:
                shape = T.even_shape(shape, fft_axes)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes, precision=precision)

        tol = get_fft_tol(g.precision)


        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array(shape, **ARRAY_GEN_CFG,
                            complex=(not real), precision=precision)

            g.psi[...] = f
            assert np.array_equal(g.psi, f)

            g.fft() 
            fk = fft2(f, axes=fft_axes)
            if not real:
                assert T.adaptive_allclose(g.psi_k, fk, **tol) # type: ignore
            else:
                fk_r = rfft2(f, axes=fft_axes)
                assert T.adaptive_allclose(g.psi_k, fk_r, **tol) # type: ignore

            g.ifft()
            f = ifft2(fk, axes=fft_axes)
            assert T.adaptive_allclose(g.psi, f, **tol) # type: ignore

            if real:
                fr = irfft2(fk_r, axes=fft_axes) # type: ignore
                assert T.adaptive_allclose(g.psi, fr, **tol) # type: ignore


