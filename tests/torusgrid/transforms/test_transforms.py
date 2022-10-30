from __future__ import annotations
from typing import Tuple, cast

from torusgrid.core.dtypes import PrecisionLike, PrecisionStr
import torusgrid.misc.testutils as T
import torusgrid.transforms as tgt
import torusgrid as tg
import pytest

import numpy as np

_cfg  = T.get_config()
config = _cfg['transforms']
gconfig = _cfg['global']

ARRAY_GEN_CFG = dict(
    min_offset=config['min_offset'], max_offset=config['max_offset'],
    min_scale=config['min_scale'], max_scale=config['max_scale']
)

def get_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['floating_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['floating_tol']))



shapes = T.gen_shapes(config['num_samples'],
                      ranks=config['ranks'], 
                      max_numel=config['max_numel'],
                      min_numel=config['min_numel'])

sizes = []
for shape in shapes:
    if np.random.rand() < 0.5: sizes.append(None)
    else: sizes.append(T.gen_random_sequence(len(shape), 
                                             min_prod=config['min_vol'], 
                                             max_prod=config['max_vol']))

reals = [[True,False][np.random.choice([0,1])] for _ in shapes]

fft_axeses = [T.gen_fft_axes(len(shape)) for shape in shapes]

axeses = [tuple(T.random_permutation(range(len(shape)))) for shape in shapes]

precs = [np.random.choice(['single', 'double', 'longdouble']) for _ in shapes]

m_factors = [T.gen_random_sequence(len(shape), 
                                   config['extend']['min_factors_prod'], 
                                   config['extend']['max_factors_prod'],
                                   integ=True) for shape in shapes]

flip_axes = [T.gen_fft_axes(len(shape)) for shape in shapes]

argvals = {'real': reals, 'shape': shapes, 'size': sizes,
           'fft_axes': fft_axeses, 'axes': axeses, 'prec': precs,
           'm_factors': m_factors, 'flip_axes': flip_axes}

def get_grid(
        real: bool, 
        shape: Tuple[int,...], size: Tuple[float,...]|None,
        fft_axes: Tuple[int,...], prec: PrecisionStr) -> tg.Grid:
    isfield = (size is not None)

    cls = T.get_class(real, isfield, 0)
    
    if size is not None:
        if T.will_overflow(size, shape, prec):
            pytest.skip(reason='float overflow')

    if real:
        shape_ = list(shape)
        if shape_[fft_axes[-1]] % 2 == 1:
            shape_[fft_axes[-1]] += 1
        shape = tuple(shape_)
    
    if isfield:
        g = cls(size, shape, fft_axes=fft_axes, precision=prec)
    else:
        g = cls(shape, fft_axes=fft_axes, precision=prec)

    return g


class TestGridTransforms:

    @T.parametrize('real', 'shape', 'size', 'fft_axes', 'axes', 'prec', argvals=argvals)
    def test_transpose(
        self, 
        real: bool, 
        shape: Tuple[int,...], size: Tuple[float,...],
        fft_axes: Tuple[int,...], axes: Tuple[int,...],
        prec: PrecisionStr):

        isfield = (size is not None)

        g = get_grid(real, shape, size, fft_axes , prec)

        g.psi[...] = T.gen_array(g.shape, **ARRAY_GEN_CFG)

        axes_map = {old_ax: new_ax for new_ax,old_ax in enumerate(axes)}
        with pytest.raises(ValueError):
            h = tgt.transpose(g, (0,0))

        h = tgt.transpose(g, axes)


        assert h.shape == tuple(g.shape[ax] for ax in axes)
        assert h.psi is not g.psi
        assert np.all(h.psi == np.transpose(g.psi, axes))
        assert h.fft_axes == tuple(axes_map[int(ax)] for ax in g.fft_axes)
        assert h._fft is None
        assert h._ifft is None
        assert h.precision is g.precision
        assert type(h) is type(g)

        if len(shape) == 2:
            if axes == (1,0):
                if g.fft_axes == (0,):
                    assert h.fft_axes == (1,)
                elif g.fft_axes == (1,):
                    assert h.fft_axes == (0,)
                elif g.fft_axes == (0,1):
                    assert h.fft_axes == (1,0)
                elif g.fft_axes == (1,0):
                    assert h.fft_axes == (0,1)
                else:
                    raise Exception


        if isfield:

            tol = get_tol(g.precision)
            g = cast(tg.Field, g)
            h = cast(tg.Field, h)

            assert np.all(h.size == [g.size[axes[ax]] for ax in range(g.rank)])

            assert np.all(h.dr == [g.dr[axes[ax]] for ax in range(g.rank)])

            r_trans = np.stack([np.transpose(g.r[axes[ax]], axes) for ax in range(g.rank)], axis=0)
            k_trans = np.stack([np.transpose(g.k[axes[ax]], axes) for ax in range(g.rank)], axis=0)

            assert np.all(h.r == r_trans)
            assert np.all(h.k == k_trans)

            assert np.isclose(g.dv, h.dv, rtol=tol['rtol1'], atol=0)


            assert T.adaptive_allclose(h.k2,
                                       np.transpose(g.k2, axes), 
                                       **tol)

    @T.parametrize('real', 'shape', 'size', 'fft_axes', 'prec', argvals=argvals)
    def test_const_like(
        self, 
        real: bool, shape: Tuple[int,...],
        size: Tuple[float,...],
        fft_axes: Tuple[int,...],
        prec: PrecisionStr):

        isfield = (size is not None)

        g = get_grid(real, shape, size, fft_axes, prec)

        if np.random.rand() < 0.5:
            k = None
        else:
            k = T.gen_array((), **ARRAY_GEN_CFG, complex=(not real), precision=prec).item()

        h = tgt.const_like(g, fill=k)

        assert T.same_meta(g, h)
        assert h.psi is not g.psi
        assert h._fft is None
        assert h._ifft is None

        if k is not None:
            assert np.all(h.psi == k)
        else:
            assert np.all(h.psi == g.psi.mean())

    @T.parametrize('real', 'shape', 'size', 'fft_axes', 'prec', 'm_factors', argvals=argvals)
    def test_extend(
        self, 
        real: bool, shape: Tuple[int,...],
        size: Tuple[float,...]|None, fft_axes: Tuple[int,...],
        prec: PrecisionStr, m_factors: Tuple[int,...]
    ):
        isfield = (size is not None)
        dtype = T.real_dtype(prec)
        
        if size is not None:
            size = [dtype(s) for s in size]

        g = get_grid(real, shape, size, fft_axes, prec)

        tol = get_tol(g.precision)

        if T.prod(m_factors) * g.numel > config['extend']['max_numel']:
            pytest.skip(reason='numel after extension too large')
        
        with pytest.raises(ValueError):
            tgt.extend(g, (1, *m_factors))

        h = tgt.extend(g, m_factors)

        assert T.same_meta_except_dimensions(g, h)
        
        assert h.shape == tuple(s * m for s,m in zip(g.shape,m_factors))
        assert h.psi is not g.psi
        assert h._fft is None
        assert h._ifft is None

        psi_extended = g.psi.copy()

        for ax,m in enumerate(m_factors):
            psi_extended = np.concatenate([psi_extended] * m, axis=ax)

        assert np.all(h.psi == psi_extended)

        if isfield:
            assert isinstance(g, tg.Field)
            assert isinstance(h, tg.Field)
            assert T.adaptive_allclose(
                    h.size, [s*m for s,m in zip(g.size, m_factors)], 
                    **tol)



    @T.parametrize('real', 'shape', 'size', 'fft_axes', 'prec', 'flip_axes', argvals=argvals)
    def test_flip( 
        self, 
        real: bool, shape: Tuple[int,...],
        size: Tuple[float,...]|None, fft_axes: Tuple[int,...],
        prec: PrecisionStr, flip_axes: Tuple[int,...]
    ):
        isfield = (size is not None)
        dtype = T.real_dtype(prec)
        
        if size is not None:
            size = [dtype(s) for s in size]

        g = get_grid(real, shape, size, fft_axes, prec)
        tol = get_tol(g.precision)

        with pytest.raises(ValueError):
            tgt.flip(g, (0,g.rank))

        with pytest.raises(ValueError):
            tgt.flip(g, (0,0,1))

        h = tgt.flip(g, flip_axes)

        assert T.same_meta(g, h)


        psi_flipped = np.flip(g.psi.copy(), flip_axes)
        assert np.array_equal(h.psi, psi_flipped)

