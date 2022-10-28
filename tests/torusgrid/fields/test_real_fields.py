from functools import partial
from typing import Tuple
import numpy as np

from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
from torusgrid.fields import RealField
import torusgrid.misc.testutils as T
from scipy.fft import rfftn, irfftn, fftn, ifftn
import pytest


_cfg = T.get_config()
config = _cfg['fields']['real']
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

shapes = T.gen_shapes(config['num_samples'], 
                      ranks=config['ranks'],
                      min_numel=config['min_numel'],
                      max_numel=config['max_numel'])

sizes = [T.gen_random_sequence(len(shape), 
                               min_prod=config['min_vol'], 
                               max_prod=config['max_vol']) 
        for shape in shapes]

fft_axeses = [T.gen_fft_axes(len(shape)) for shape in shapes]

precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]

argvals = dict(size=sizes, shape=shapes, fft_axes=fft_axeses, precision=precisions)


class TestRealFields:

    @T.parametrize('size', 'shape', 'fft_axes', 'precision', argvals=argvals)
    def test_props(self, 
                   size,
                   shape: Tuple[int,...], 
                   fft_axes: Tuple[int,...], 
                   precision: PrecisionStr): 

        last_fft_axis = fft_axes[-1]
        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')

        dtype = T.real_dtype(precision)
        size = [dtype(s) for s in size]


        if shape[last_fft_axis] % 2 == 1:
            with pytest.warns(UserWarning):
                g = RealField(size, shape, fft_axes=fft_axes, precision=precision)

            assert g.shape[g.last_fft_axis] == shape[last_fft_axis] + 1
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)
        else:
            g = RealField(size, shape, fft_axes=fft_axes, precision=precision)

        

        tol = get_tol(g.precision)

        shape_k_ = list(shape)
        shape_k_[last_fft_axis] //= 2
        shape_k_[last_fft_axis] += 1
        shape_k = tuple(shape_k_)
        
        with pytest.warns(np.ComplexWarning):
            g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

        assert g.shape == shape
        assert g.psi.shape == shape
        assert g.psi_k.shape == shape_k
        assert g.rank == len(shape)

        assert np.array_equal(g.size, size)

        assert g.numel == T.prod(shape)
        assert g.r.shape[0] == len(shape)
        assert g.r.shape[1:] == shape

        assert g.fft_axes == fft_axes
        assert g.last_fft_axis == fft_axes[-1]

        assert g.shape_k == g.k.shape[1:]

        assert np.size(g.r) == g.rank * T.prod(shape)
        assert np.isclose(g.volume, 
                          T.prod(g.size.tolist()), 
                          rtol=tol['rtol1'], atol=0)
        
        r = []
        for l, n in zip(size, shape):
            r.append(np.linspace(0, l, n+1, dtype=dtype)[:-1])

        r = np.stack(np.meshgrid(*r, indexing='ij'), axis=0)
        assert T.adaptive_allclose(r, g.r, **tol)

        k = []
        for axis, (L, N) in enumerate(zip(size, shape)):
            dk = np.pi*2 / dtype(L)
            M = N // 2
            _k = [i*dk for i in range(M)] + [i*dk for i in range(-(N-M), 0)]

            if axis == last_fft_axis:
                assert N % 2 == 0
                _k = [i*dk for i in range(M+1)]

            k.append(np.array(_k))

        k = np.stack(np.meshgrid(*k, indexing='ij'), axis=0)

        assert T.adaptive_allclose(k, g.k, **tol)
        k2 = np.sum(k**2, axis=0)

        assert T.adaptive_allclose(k2, g.k2, **tol)
        assert T.adaptive_allclose(k2**2, g.k4, **tol)

        assert g._fft is None
        assert g._ifft is None
        assert g.isreal


    @T.parametrize('size', 'shape', 'fft_axes', 'precision', argvals=argvals)
    def test_copy(self, size, shape, fft_axes, precision):

        if shape[fft_axes[-1]] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)
        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')


        g = RealField(size, shape, fft_axes=fft_axes, precision=precision)
        tol = get_tol(g.precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

        h = g.copy()

        assert np.array_equal(g.psi, h.psi)
        assert g.psi is not h.psi
        assert g.psi_k is not h.psi_k

        assert T.same_meta(g, h, **tol)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

        assert not np.array_equal(g.psi, h.psi)


    @T.parametrize('size', 'shape', 'fft_axes', 'precision', argvals=argvals)
    def test_set_size(self, size, shape, fft_axes, precision):
        last_fft_axis = fft_axes[-1]
        if shape[fft_axes[-1]] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)



        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')


        g = RealField(size, shape, fft_axes=fft_axes, precision=precision)
        tol = get_tol(g.precision)
        dtype = T.real_dtype(precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)
        
        for _ in range(config['num_setsize_trials']):
            size_ = T.gen_random_sequence(len(size), 
                                          min_prod=config['min_vol'],
                                          max_prod=config['max_vol'])

            size_ = [dtype(s) for s in size_]

            if T.will_overflow(size_, shape, precision): continue

            g.set_size(size_)

            assert T.adaptive_allclose(g.size, size_, **tol) # type: ignore

            r = []
            for l, n in zip(size_, shape):
                r.append(np.linspace(0, l, n+1, dtype=dtype)[:-1])

            r = np.stack(np.meshgrid(*r, indexing='ij'), axis=0)
            assert T.adaptive_allclose(r, g.r, **tol)

            k = []
            for axis, (L, N) in enumerate(zip(size_, shape)):
                dk = np.pi*2 / dtype(L)
                M = N // 2
                _k = [i*dk for i in range(M)] + [i*dk for i in range(-(N-M), 0)]

                if axis == last_fft_axis:
                    assert N % 2 == 0
                    _k = [i*dk for i in range(M+1)]

                k.append(np.array(_k))
            k = np.stack(np.meshgrid(*k, indexing='ij'), axis=0)
            assert T.adaptive_allclose(k, g.k, **tol)
            k2 = np.sum(k**2, axis=0)

            assert T.adaptive_allclose(k2, g.k2, **tol)
            assert T.adaptive_allclose(k2**2, g.k4, **tol)


    @T.parametrize('size', 'shape', 'fft_axes', 'precision', argvals=argvals)
    def test_precision(self, size, shape, fft_axes, precision):
        """
        Test whether the field has attributes with the appropriate precision
        """
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)

        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')


        g = RealField(size, shape, 
                      fft_axes=fft_axes,
                      precision=precision)

        precision = precision.upper()
      
        assert g.psi.dtype == T.real_dtype(precision)
        assert g.psi.dtype != T.complex_dtype(precision)
      
        assert g.psi_k.dtype == T.complex_dtype(precision)
      
        assert g.psi.nbytes == g.numel * T.nbytes(precision) * 1
       
        assert g.psi_k.nbytes == T.prod(g.shape_k) * T.nbytes(precision) * 2
       
        assert g.size.dtype == T.real_dtype(precision)
        assert g.r.dtype == T.real_dtype(precision)
        assert g.k.dtype == T.real_dtype(precision)

        assert g.k2.dtype == T.real_dtype(precision)
        assert g.k4.dtype == T.real_dtype(precision)
        assert g.k6.dtype == T.real_dtype(precision)

        assert g.precision is FloatingPointPrecision.cast(precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)
        assert g.psi.dtype == T.real_dtype(precision)


    @pytest.mark.slow 
    @T.parametrize('size', 'shape', 'fft_axes', 'precision', argvals=argvals)
    def test_fft(self, size, shape, fft_axes, precision):
        """Test FFT"""
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)

        if T.will_overflow(size, shape, precision):
            pytest.skip(msg='float overflow')

        g = RealField(size, shape, fft_axes=fft_axes, precision=precision)
        tol = get_fft_tol(g.precision)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

            g.psi[...] = f

            assert np.array_equal(g.psi, f)

            g.fft()
            fk = rfftn(f, axes=fft_axes)
            fk_ = fftn(f, axes=fft_axes)
            assert T.adaptive_allclose(g.psi_k, fk, **tol) # type: ignore

            f_ = irfftn(fk, axes=fft_axes)
            f__ = ifftn(fk_, axes=fft_axes)

            assert T.adaptive_allclose(g.psi, f_, **tol) # type: ignore
            assert T.adaptive_allclose(g.psi, f__, **tol) # type: ignore


