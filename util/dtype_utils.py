# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import OrderedDict
from datetime import datetime
from distutils.version import StrictVersion
import numpy as np
from numpy import (
    datetime64,
    dtype,
    # 创建nan变量
    isnan,
    nan
)
from toolz import flip

numpy_version = StrictVersion(np.__version__)

uint8_dtype = dtype('uint8')
bool_dtype = dtype('bool')

uint32_dtype = dtype('uint32')
uint64_dtype = dtype('uint64')
int64_dtype = dtype('int64')

float32_dtype = dtype('float32')
float64_dtype = dtype('float64')

complex128_dtype = dtype('complex128')

datetime64D_dtype = dtype('datetime64[D]')
datetime64ns_dtype = dtype('datetime64[ns]')

make_datetime64ns = flip(datetime64, 'ns')
make_datetime64D = flip(datetime64, 'D')

NaTmap = {
    dtype('datetime64[%s]' % unit): datetime64('NaT', unit)
    for unit in ('ns', 'us', 'ms', 's', 'm', 'D')
}


def NaT_for_dtype(dtype):
    """Retrieve NaT with the same units as ``dtype``.

    Parameters
    ----------
    dtype : dtype-coercable
        The dtype to lookup the NaT value for.

    Returns
    -------
    NaT : dtype
        The NaT value for the given dtype.
    """
    return NaTmap[np.dtype(dtype)]


NaTns = NaT_for_dtype(datetime64ns_dtype)
NaTD = NaT_for_dtype(datetime64D_dtype)


_FILLVALUE_DEFAULTS = {
    bool_dtype: False,
    float32_dtype: nan,
    float64_dtype: nan,
    datetime64ns_dtype: NaTns,
    object_dtype: None,
}

INT_DTYPES_BY_SIZE_BYTES = OrderedDict([
    (1, dtype('int8')),
    (2, dtype('int16')),
    (4, dtype('int32')),
    (8, dtype('int64')),
])

UNSIGNED_INT_DTYPES_BY_SIZE_BYTES = OrderedDict([
    (1, dtype('uint8')),
    (2, dtype('uint16')),
    (4, dtype('uint32')),
    (8, dtype('uint64')),
])


def default_missing_value_for_dtype(dtype):
    """
    Get the default fill value for `dtype`.
    """
    try:
        return _FILLVALUE_DEFAULTS[dtype]
    except KeyError:
        raise NoDefaultMissingValue(
            "No default value registered for dtype %s." % dtype
        )


def int_dtype_with_size_in_bytes(size):
    try:
        return INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError:
        raise ValueError("No integral dtype whose size is %d bytes." % size)


def unsigned_int_dtype_with_size_in_bytes(size):
    try:
        return UNSIGNED_INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError:
        raise ValueError(
            "No unsigned integral dtype whose size is %d bytes." % size
        )


def make_kind_check(python_types, numpy_kind):
    """
    Make a function that checks whether a scalar or array is of a given kind
    (e.g. float, int, datetime, timedelta).
    """
    def check(value):
        if hasattr(value, 'dtype'):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)
    return check


is_float = make_kind_check(float, 'f')
is_int = make_kind_check(int, 'i')
is_datetime = make_kind_check(datetime, 'M')
is_object = make_kind_check(object, 'O')


def coerce_to_dtype(dtype, value):
    """
    Make a value with the specified numpy dtype.

    Only datetime64[ns] and datetime64[D] are supported for datetime dtypes.
    """
    name = dtype.name
    if name.startswith('datetime64'):
        if name == 'datetime64[D]':
            return make_datetime64D(value)
        elif name == 'datetime64[ns]':
            return make_datetime64ns(value)
        else:
            raise TypeError(
                "Don't know how to coerce values of dtype %s" % dtype
            )
    return dtype.type(value)


# Sentinel value that isn't NaT.
_notNaT = make_datetime64D(0)
iNaT = int(NaTns.view(int64_dtype))
assert iNaT == NaTD.view(int64_dtype), "iNaTns != iNaTD"


def isnat(obj):
    """
    Check if a value is np.NaT.
    """
    if obj.dtype.kind not in ('m', 'M'):
        raise ValueError("%s is not a numpy datetime or timedelta")
    return obj.view(int64_dtype) == iNaT


def is_missing(data, missing_value):
    """
    Generic is_missing function that handles NaN and NaT.
    """
    if is_float(data) and isnan(missing_value):
        return isnan(data)
    elif is_datetime(data) and isnat(missing_value):
        return isnat(data)
    return data == missing_value
