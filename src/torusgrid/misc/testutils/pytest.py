from __future__ import annotations
from typing import Tuple, Any, Dict
import numpy as np
import pytest


# def get_test_id(size, shape, fft_axes)
#     """Test IDs"""
#     return f'size={float_tup_repr(size)} shape={int_tup_repr(shape)} fft={int_tup_repr(fft_axes)}'


def float_tup_repr(tup, fmt: str='.3e'):
    """
    Representation of float tuple
    """
    return '(' + ','.join([f'{s:{fmt}}' for s in tup]) + ')'


def int_tup_repr(tup: Tuple[int,...]):
    """
    Representation of int tuple
    """
    return '(' + ','.join([str(s) for s in tup]) + ')'


def param_repr(arg: Any) -> str:
    """
    Return the representation of a test parameter 
    """
    if isinstance(arg, tuple):
        if isinstance(arg[0], (int,np.integer)):
            return int_tup_repr(arg)
        elif isinstance(arg[0], (float,np.floating)):
            return float_tup_repr(arg)
    return repr(arg)


def params_repr(argtup: Tuple[Any,...], *, delim='|'):
    """
    Return the representation of a set of test parameters
    """
    return delim.join(param_repr(a) for a in argtup)


def parametrize(*argnames: str, argvals: Dict[str, Any]):
    """
    A wrapper of pytest.parametrize
    """
    def wrapper(f):
        _used_argvals = [_ for _ in zip(*[argvals[argname] for argname in argnames])]
        _test_ids = [params_repr(argtup) for argtup in _used_argvals]


        return pytest.mark.parametrize(
                list(argnames), _used_argvals,
                ids=_test_ids)(f)
    return wrapper



