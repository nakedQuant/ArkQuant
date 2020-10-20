# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from decimal import Decimal
import math, numpy as np, pandas as pd
from numpy import isnan


# transform nan to specific value
def nan_proc(x):
    np.nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None)


def round_if_near_integer(a, epsilon=1e-4):
    """
    Round a to the nearest integer if that integer is within an epsilon
    of a.
    """
    if abs(a - round(a)) <= epsilon:
        return round(a)
    else:
        return a


def consistent_round(val):
    if (val % 1) >= 0.5:
        return np.ceil(val)
    else:
        return round(val)


def tolerant_equals(a, b, atol=10e-7, rtol=10e-7, equal_nan=False):
    """Check if a and b are equal with some tolerance.

    Parameters
    ----------
    a, b : float
        The floats to check for equality.
    atol : float, optional
        The absolute tolerance.
    rtol : float, optional
        The relative tolerance.
    equal_nan : bool, optional
        Should NaN compare equal?

    See Also
    --------
    numpy.isclose

    Notes
    -----
    This function is just a scalar version of numpy.isclose for performance.
    See the docstring of ``isclose`` for more information about ``atol`` and
    ``rtol``.
    """
    if equal_nan and isnan(a) and isnan(b):
        return True
    return math.fabs(a - b) <= (atol + rtol * math.fabs(b))


# 小数点位数
def number_of_decimal_places(n):
    """
    Compute the number of decimal places in a number.

    Examples
    --------
    >>> number_of_decimal_places('3.14')
    2
    """
    decimal = Decimal(str(n))
    return -decimal.as_tuple().exponent


def _gen_unzip(it, elem_len):
    """Helper for unzip which checks the lengths of each element in it.
    Parameters
    ----------
    it : iterable[tuple]
        An iterable of tuples. ``unzip`` should map ensure that these are
        already tuples.
    elem_len : int or None
        The expected element length. If this is None it is infered from the
        length of the first element.
    """
    elem = next(it)
    first_elem_len = len(elem)

    if elem_len is not None and elem_len != first_elem_len:
        raise ValueError(
            'element at index 0 was length %d, expected %d' % (
                first_elem_len,
                elem_len,
            )
        )
    else:
        elem_len = first_elem_len

    yield elem
    for n, elem in enumerate(it, 1):
        if len(elem) != elem_len:
            raise ValueError(
                'element at index %d was length %d, expected %d' % (
                    n,
                    len(elem),
                    elem_len,
                ),
            )
        yield elem


def vectorized_is_element(array, choices):
    # numpy.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
    return np.vectorize(choices.__contains__, otypes=[bool])(array)


def measure_volatity(data):
    if isinstance(data, pd.DataFrame):
        std = (data['high'] - data['low']).std()
    elif isinstance(data, np.array):
        std = np.std(data)
    return std

