#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from decimal import Decimal
import math

from numpy import isnan

def round_if_near_integer(a, epsilon=1e-4):
    """
    Round a to the nearest integer if that integer is within an epsilon
    of a.
    """
    if abs(a - round(a)) <= epsilon:
        return round(a)
    else:
        return a

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
    >>> number_of_decimal_places(1)
    0
    >>> number_of_decimal_places(3.14)
    2
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

def dzip_exact(*dicts):
    """
    >>> result = dzip_exact({'a': 1, 'b': 2}, {'a': 3, 'b': 4})
    >>> result == {'a': (1, 3), 'b': (2, 4)}
    True
    """
    if not same(*map(viewkeys, dicts)):
        raise ValueError(
            "dict keys not all equal:\n\n%s" % _format_unequal_keys(dicts)
        )
    return {k: tuple(d[k] for d in dicts) for k in dicts[0]}

def invert(d):
    """
    >>> invert({'a': 1, 'b': 2, 'c': 1})  # doctest: +SKIP
    {1: {'a', 'c'}, 2: {'b'}}
    """
    out = {}
    for k, v in iteritems(d):
        try:
            out[v].add(k)
        except KeyError:
            out[v] = {k}
    return out


def keysorted(d):
    """Get the items from a dict, sorted by key.

    Example
    -------
    >>> keysorted({'c': 1, 'b': 2, 'a': 3})
    [('a', 3), ('b', 2), ('c', 1)]
    """
    return sorted(iteritems(d), key=itemgetter(0))
