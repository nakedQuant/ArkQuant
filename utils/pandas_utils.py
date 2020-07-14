"""
Utilities for working with pandas objects.
"""
from contextlib import contextmanager
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
from distutils.version import StrictVersion

from warning import ignore_pandas_nan_categorical_warning

pandas_version = StrictVersion(pd.__version__)
new_pandas = pandas_version >= StrictVersion('0.19')
skip_pipeline_new_pandas = \
    'Pipeline categoricals are not yet compatible with pandas >=0.19'

if pandas_version >= StrictVersion('0.20'):
    def normalize_date(dt):
        """
        Normalize datetime.datetime value to midnight. Returns datetime.date as
        a datetime.datetime at midnight

        Returns
        -------
        normalized : datetime.datetime or Timestamp
        """
        return dt.normalize()
else:
    from pandas.tseries.tools import normalize_date


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


def find_in_sorted_index(dts, dt):
    """
    Find the index of ``dt`` in ``dts``.

    This function should be used instead of `dts.get_loc(dt)` if the index is
    large enough that we don't want to initialize a hash table in ``dts``. In
    particular, this should always be used on minutely trading calendars.

    Parameters
    ----------
    dts : pd.DatetimeIndex
        Index in which to look up ``dt``. **Must be sorted**.
    dt : pd.Timestamp
        ``dt`` to be looked up.

    Returns
    -------
    ix : int
        Integer index such that dts[ix] == dt.

    Raises
    ------
    KeyError
        If dt is not in ``dts``.
    """
    # searchsorted(a,v,side = 'left',sorter = None) left:a[i-1] < v <= a[i] ; right a[i-1] <= v <a[i]
    ix = dts.searchsorted(dt)
    if ix == len(dts) or dts[ix] != dt:
        raise LookupError("{dt} is not in {dts}".format(dt=dt, dts=dts))
    return ix


def nearest_unequal_elements(dts, dt):
    """
    Find values in ``dts`` closest but not equal to ``dt``.

    Returns a pair of (last_before, first_after).

    When ``dt`` is less than any element in ``dts``, ``last_before`` is None.
    When ``dt`` is greater any element in ``dts``, ``first_after`` is None.

    ``dts`` must be unique and sorted in increasing order.

    Parameters
    ----------
    dts : pd.DatetimeIndex
        Dates in which to search.
    dt : pd.Timestamp
        Date for which to find bounds.
    """
    if not dts.is_unique:
        raise ValueError("dts must be unique")

    if not dts.is_monotonic_increasing:
        raise ValueError("dts must be sorted in increasing order")

    if not len(dts):
        return None, None

    sortpos = dts.searchsorted(dt, side='left')
    try:
        sortval = dts[sortpos]
    except IndexError:
        # dt is greater than any value in the array.
        return dts[-1], None

    if dt < sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos
    elif dt == sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos + 1
    else:
        lower_ix = sortpos
        upper_ix = sortpos + 1

    lower_value = dts[lower_ix] if lower_ix >= 0 else None
    upper_value = dts[upper_ix] if upper_ix < len(dts) else None

    return lower_value, upper_value


def categorical_df_concat(df_list, inplace=False):
    """
    Prepare list of pandas DataFrames to be used as input to pd.concat.
    Ensure any columns of type 'category' have the same categories across each
    dataframe.

    Parameters
    ----------
    df_list : list
        List of dataframes with same columns.
    inplace : bool
        True if input list can be modified. Default is False.

    Returns
    -------
    concatenated : df
        Dataframe of concatenated list.
    """

    if not inplace:
        df_list = deepcopy(df_list)

    # Assert each dataframe has the same columns/dtypes
    df = df_list[0]
    if not all([(df.dtypes.equals(df_i.dtypes)) for df_i in df_list[1:]]):
        raise ValueError("Input DataFrames must have the same columns/dtypes.")

    categorical_columns = df.columns[df.dtypes == 'category']
    # 基于类别获取不同DataFrame分类 --- 重新分类
    for col in categorical_columns:
        new_categories = sorted(
            set().union(
                *(frame[col].cat.categories for frame in df_list)
            )
        )

        with ignore_pandas_nan_categorical_warning():
            for df in df_list:
                df[col].cat.set_categories(new_categories, inplace=True)

    return pd.concat(df_list)


def check_indexes_all_same(indexes, message="Indexes are not equal."):
    """Check that a list of Index objects are all equal.

    Parameters
    ----------
    indexes : iterable[pd.Index]
        Iterable of indexes to check.

    Raises
    ------
    ValueError
        If the indexes are not all the same.
    """
    iterator = iter(indexes)
    first = next(iterator)
    for other in iterator:
        same = (first == other)
        if not same.all():
            #返回非0元素的位置
            bad_loc = np.flatnonzero(~same)[0]
            raise ValueError(
                "{}\nFirst difference is at index {}: "
                "{} != {}".format(
                    message, bad_loc, first[bad_loc], other[bad_loc]
                ),
            )
