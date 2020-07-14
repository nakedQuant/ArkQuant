"""
Utilities for working with numpy arrays.
"""
from collections import OrderedDict
from datetime import datetime
from distutils.version import StrictVersion
from warnings import (
    catch_warnings,
    filterwarnings,
)

import numpy as np
from numpy import (
    array_equal,
    broadcast,
    busday_count,
    datetime64,
    diff,
    dtype,
    empty,
    flatnonzero,
    hstack,
    #创建nan变量
    isnan,
    nan,
    vectorize,
    where
)
from numpy.lib.stride_tricks import as_strided
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

object_dtype = dtype('O')
# We use object arrays for strings.
categorical_dtype = object_dtype

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


class NoDefaultMissingValue(Exception):
    pass


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


def repeat_first_axis(array, count):
    """
    Restride `array` to repeat `count` times along the first axis.

    Parameters
    ----------
    array : np.array
        The array to restride.
    count : int
        Number of times to repeat `array`.

    Returns
    -------
    result : array
        Array of shape (count,) + array.shape, composed of `array` repeated
        `count` times along the first axis.

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(3); a
    array([0, 1, 2])
    >>> repeat_first_axis(a, 2)
    array([[0, 1, 2],
           [0, 1, 2]])
    >>> repeat_first_axis(a, 4)
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]])

    Notes
    ----
    The resulting array will share memory with `array`.  If you need to assign
    to the input or output, you should probably make a copy first.

    See Also
    --------
    repeat_last_axis
    """
    return as_strided(array, (count,) + array.shape, (0,) + array.strides)


def repeat_last_axis(array, count):
    """
    Restride `array` to repeat `count` times along the last axis.

    Parameters
    ----------
    array : np.array
        The array to restride.
    count : int
        Number of times to repeat `array`.

    Returns
    -------
    result : array
        Array of shape array.shape + (count,) composed of `array` repeated
        `count` times along the last axis.

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(3); a
    array([0, 1, 2])
    >>> repeat_last_axis(a, 2)
    array([[0, 0],
           [1, 1],
           [2, 2]])
    >>> repeat_last_axis(a, 4)
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [2, 2, 2, 2]])

    Notes
    ----
    The resulting array will share memory with `array`.  If you need to assign
    to the input or output, you should probably make a copy first.

    See Also
    --------
    repeat_last_axis
    """
    return as_strided(array, array.shape + (count,), array.strides + (0,))


def rolling_window(array, length):
    """
    Restride an array of shape

        (X_0, ... X_N)

    into an array of shape

        (length, X_0 - length + 1, ... X_N)

    where each slice at index i along the first axis is equivalent to

        result[i] = array[length * i:length * (i + 1)]

    Parameters
    ----------
    array : np.ndarray
        The base array.
    length : int
        Length of the synthetic first axis to generate.

    Returns
    -------
    out : np.ndarray

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(25).reshape(5, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> rolling_window(a, 2)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    <BLANKLINE>
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])
    """
    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] <= length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = (orig_shape[0] - length + 1)
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    return as_strided(array, new_shape, new_strides)


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
    return (data == missing_value)


def busday_count_mask_NaT(begindates, enddates, out=None):
    """
    Simple of numpy.busday_count that returns `float` arrays rather than int
    arrays, and handles `NaT`s by returning `NaN`s where the inputs were `NaT`.

    Doesn't support custom weekdays or calendars, but probably should in the
    future.

    See Also
    --------
    np.busday_count
    """
    if out is None:
        out = empty(broadcast(begindates, enddates).shape, dtype=float)

    beginmask = isnat(begindates)
    endmask = isnat(enddates)

    out = busday_count(
        # Temporarily fill in non-NaT values.
        where(beginmask, _notNaT, begindates),
        where(endmask, _notNaT, enddates),
        out=out,
    )

    # Fill in entries where either comparison was NaT with nan in the output.
    out[beginmask | endmask] = nan
    return out


class WarningContext(object):
    """
    Re-usable contextmanager for contextually managing warnings.
    """
    def __init__(self, *warning_specs):
        self._warning_specs = warning_specs
        self._catchers = []

    def __enter__(self):
        catcher = catch_warnings()
        catcher.__enter__()
        self._catchers.append(catcher)
        for args, kwargs in self._warning_specs:
            filterwarnings(*args, **kwargs)
        return self

    def __exit__(self, *exc_info):
        catcher = self._catchers.pop()
        return catcher.__exit__(*exc_info)


def ignore_nanwarnings():
    """
    Helper for building a WarningContext that ignores warnings from numpy's
    nanfunctions.
    """
    return WarningContext(
        (
            ('ignore',),
            {'category': RuntimeWarning, 'module': 'numpy.lib.nanfunctions'},
        )
    )


def vectorized_is_element(array, choices):
    """
    Check if each element of ``array`` is in choices.

    Parameters
    ----------
    array : np.ndarray
    choices : object
        Object implementing __contains__.

    Returns
    -------
    was_element : np.ndarray[bool]
        Array indicating whether each element of ``array`` was in ``choices``.
    """
    return vectorize(choices.__contains__, otypes=[bool])(array)


def as_column(a):
    """
    Convert an array of shape (N,) into an array of shape (N, 1).

    This is equivalent to `a[:, np.newaxis]`.

    Parameters
    ----------
    a : np.ndarray

    Example
    -------
    >>> import numpy as np
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> as_column(a)
    array([[0],
           [1],
           [2],
           [3],
           [4]])
    >>> as_column(a).shape
    (5, 1)
    """
    if a.ndim != 1:
        raise ValueError(
            "as_column expected an 1-dimensional array, "
            "but got an array of shape %s" % a.shape
        )
    return a[:, None]


def changed_locations(a, include_first):
    """
    Compute indices of values in ``a`` that differ from the previous value.

    Parameters
    ----------
    a : np.ndarray
        The array on which to indices of change.
    include_first : bool
        Whether or not to consider the first index of the array as "changed".

    Example
    -------
    >>> import numpy as np
    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=False)
    array([2, 4])

    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=True)
    array([0, 2, 4])
    """
    if a.ndim > 1:
        raise ValueError("indices_of_changed_values only supports 1D arrays.")
    indices = flatnonzero(diff(a)) + 1

    if not include_first:
        return indices

    return hstack([[0], indices])


def compare_datetime_arrays(x, y):
    """
    Compare datetime64 ndarrays, treating NaT values as equal.
    """

    return array_equal(x.view('int64'), y.view('int64'))


from numpy.lib import apply_along_axis
from pandas import qcut


def quantiles(data, nbins_or_partition_bounds):
    """
    Compute rowwise array quantiles on an input.
    """
    return apply_along_axis(
        qcut,
        1,
        data,
        q=nbins_or_partition_bounds, labels=False,
    )

def naive_grouped_rowwise_apply(data,
                                group_labels,
                                func,
                                func_args=(),
                                out=None):
    """
    Simple implementation of grouped row-wise function application.

    Parameters
    ----------
    data : ndarray[ndim=2]
        Input array over which to apply a grouped function.
    group_labels : ndarray[ndim=2, dtype=int64]
        Labels to use to bucket inputs from array.
        Should be the same shape as array.
    func : function[ndarray[ndim=1]] -> function[ndarray[ndim=1]]
        Function to apply to pieces of each row in array.
    func_args : tuple
        Additional positional arguments to provide to each row in array.
    out : ndarray, optional
        Array into which to write output.  If not supplied, a new array of the
        same shape as ``data`` is allocated and returned.

    Examples
    --------
    >>> data = np.array([[1., 2., 3.],
    ...                  [2., 3., 4.],
    ...                  [5., 6., 7.]])
    >>> labels = np.array([[0, 0, 1],
    ...                    [0, 1, 0],
    ...                    [1, 0, 2]])
    >>> naive_grouped_rowwise_apply(data, labels, lambda row: row - row.min())
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  2.],
           [ 0.,  0.,  0.]])
    """
    if out is None:
        out = np.empty_like(data)

    for (row, label_row, out_row) in zip(data, group_labels, out):
        for label in np.unique(label_row):
            locs = (label_row == label)
            out_row[locs] = func(row[locs], *func_args)
    return out

def is_sorted_ascending(a):
    """Check if a numpy array is sorted."""
    return (np.fmax.accumulate(a) <= a).all()


def ffill_across_cols(df, columns, name_map):
    """
    Forward fill values in a DataFrame with special logic to handle cases
    that pd.DataFrame.ffill cannot and cast columns to appropriate types.
    """
    df.ffill(inplace=True)

    # Fill in missing values specified by each column. This is made
    # significantly more complex by the fact that we need to work around
    # two pandas issues:

    # 1) When we have sids, if there are no records for a given sid for any
    #    dates, pandas will generate a column full of NaNs for that sid.
    #    This means that some of the columns in `dense_output` are now
    #    float instead of the intended dtype, so we have to coerce back to
    #    our expected type and convert NaNs into the desired missing value.

    # 2) DataFrame.ffill assumes that receiving None as a fill-value means
    #    that no value was passed.  Consequently, there's no way to tell
    #    pandas to replace NaNs in an object column with None using fillna,
    #    so we have to roll our own instead using df.where.
    for column in columns:
        column_name = name_map[column.name]
        # Special logic for strings since `fillna` doesn't work if the
        # missing value is `None`.
        if column.dtype == categorical_dtype:
            df[column_name] = df[
                column.name
            ].where(pd.notnull(df[column_name]),
                    column.missing_value)
        else:
            df[column_name] = df[
                column_name
            ].fillna(column.missing_value).astype(column.dtype)


# out = np.full((len(all_dates), len(all_sids)), -1, dtype=np.int64)
#
# sid_ixs = all_sids.searchsorted(event_sids)
# # side='right' here ensures that we include the event date itself
# # if it's in all_dates.
# dt_ixs = all_dates.searchsorted(event_dates, side='right')
# ts_ixs = data_query_cutoff.searchsorted(event_timestamps, side='right')

"""
>>> from scipy.stats import rankdata
>>> rankdata([0, 2, 3, 2])
array([ 1. ,  2.5,  4. ,  2.5])
>>> rankdata([0, 2, 3, 2], method='min')
array([ 1,  2,  4,  2])
>>> rankdata([0, 2, 3, 2], method='max')
array([ 1,  3,  4,  3])
>>> rankdata([0, 2, 3, 2], method='dense')
array([ 1,  2,  3,  2])
>>> rankdata([0, 2, 3, 2], method='ordinal')
array([ 1,  2,  4,  3])
"""