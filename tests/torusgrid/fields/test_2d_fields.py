from __future__ import annotations
from typing import Tuple
import pytest
from torusgrid.fields import ComplexField2D, RealField2D

from scipy.fft import fft2, ifft2,rfft2, irfft2
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

MIN_VOL = 1e-8
MAX_VOL = 1e8

TOL = dict(rtol=FFT_RTOL, atol_factor=FFT_ATOL_FACTOR)

ARRAY_GEN_CFG = dict(
    min_offset=MIN_OFFSET, max_offset=MAX_OFFSET,
    min_scale=MIN_SCALE, max_scale=MAX_SCALE
)

shapes = T.gen_shapes(NUM_SHAPES, ranks=[2], 
                      min_log_numel=np.log(MIN_NUMEL),
                      max_log_numel=np.log(MAX_NUMEL))

sizes = [T.gen_random_sequence(2, MIN_VOL, MAX_VOL) for _ in shapes]

_test_ids = [f'shape={T.int_tup_repr(shape)} size={T.float_tup_repr(size)}'
             for shape,size in zip(shapes,sizes)]

fft_axeses = [None, (0,1), (0,), (1,), (1,0)]

def parametrize(f):
    f = pytest.mark.parametrize(
            ['shape', 'size'], zip(shapes,sizes), ids=_test_ids)(f)

    f = pytest.mark.parametrize(
            'fft_axes', fft_axeses, ids=[str(fft_ax) for fft_ax in fft_axeses]
            )(f)

    f = pytest.mark.parametrize('real', [True, False], ids=['real', 'complex'])(f)
    
    return f


class TestField2D:

    @parametrize
    def test_props(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real: bool
    ):
        ...

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealField2D(*size, *shape, fft_axes=fft_axes)
                assert g.shape[last_fft_axis] == shape[last_fft_axis] + 1
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)
            else:
                g = RealField2D(*size, *shape, fft_axes=fft_axes)

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

        r = []
        k = []
        for ax in [0,1]:
            L, N = size[ax], shape[ax]
            r.append(np.linspace(0, L, N+1)[:-1])
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

        assert T.adaptive_atol_allclose(g.r, r, **TOL)
        assert T.adaptive_atol_allclose(g.k, k, **TOL)
        assert g.r.shape == r.shape
        assert g.k.shape == k.shape
        assert T.adaptive_atol_allclose(g.k2, k2, **TOL)
        assert T.adaptive_atol_allclose(g.k4, k2**2, **TOL)
        assert T.adaptive_atol_allclose(g.k6, k2**3, **TOL)


    @parametrize
    def test_copy(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real: bool
    ):
        ...
        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real))

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.shape == h.shape
        assert g.rank == h.rank
        assert g.nx == h.nx
        assert g.ny == h.ny

        assert g.lx == h.lx
        assert g.ly == h.ly

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

        assert np.array_equal(g.dr, h.dr)

        assert g.dv == h.dv

    @parametrize 
    def test_set_size(
        self,
        size: Tuple[float,float],
        shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real: bool
    ):
        ...

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes)

        for _ in range(NUM_SETSIZE_TRIALS):
            lx_, ly_ = T.gen_random_sequence(
                    2, min_prod=MIN_VOL, max_prod=MAX_VOL)

            if np.random.rand() < 0.5:
                g.set_size(lx_, ly_)
            else:
                g.set_size((lx_, ly_))

            assert T.adaptive_atol_allclose(g.size, (lx_, ly_), **TOL) # type: ignore

            r = []
            k = []
            for ax, L, N in zip([0,1], (lx_, ly_), shape):
                r.append(np.linspace(0, L, N+1)[:-1])
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

            assert T.adaptive_atol_allclose(g.r, r, **TOL)
            assert T.adaptive_atol_allclose(g.k, k, **TOL)
            assert T.adaptive_atol_allclose(g.k2, k2, **TOL)
            assert T.adaptive_atol_allclose(g.k4, k2**2, **TOL)
            assert T.adaptive_atol_allclose(g.k6, k2**3, **TOL)


    @pytest.mark.slow
    @parametrize
    def test_fft(
        self, 
        size: Tuple[float,float], shape: Tuple[int,int],
        fft_axes: Tuple[int,...]|None,
        real: bool
    ):
        ...

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexField2D(*size, *shape, 
                               fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)

            g = RealField2D(*size, *shape,
                            fft_axes=fft_axes)


        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = T.gen_array(shape, **ARRAY_GEN_CFG,
                            complex=(not real))

            g.psi[...] = f
            assert np.array_equal(g.psi, f)

            g.fft() 
            fk = fft2(f, axes=fft_axes)
            if not real:
                assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore
            else:
                fk_r = rfft2(f, axes=fft_axes)
                assert T.adaptive_atol_allclose(g.psi_k, fk_r, **TOL) # type: ignore

            g.ifft()
            f = ifft2(fk, axes=fft_axes)
            assert T.adaptive_atol_allclose(g.psi, f, **TOL) # type: ignore

            if real:
                fr = irfft2(fk_r, axes=fft_axes) # type: ignore
                assert T.adaptive_atol_allclose(g.psi, fr, **TOL) # type: ignore


