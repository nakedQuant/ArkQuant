# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import os, inspect, numpy as np, pandas as pd, logging
from copy import deepcopy
from util.wrapper import  ignore_pandas_nan_categorical_warning


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
    return np.vectorize(choices.__contains__, otypes=[bool])(array)


def getargspec(f):
    full_argspec = inspect.getfullargspec(f)
    return inspect.ArgSpec(
        args = full_argspec.args,
        varargs = full_argspec.varargs,
        keywords = full_argspec.varkw,
        defaults = full_argspec.defaults)


def signature():
    """
        sig = signature(foo)
        str(sig)
        #'(a, *, b:int, **kwargs)'
        str(sig.parameters['b'])
        #'b:int'
        sig.parameters['b'].annotation
        #<class 'int'>
    """


def display():
    """
        pandas DataFrame表格最大显示行数
        pd.options.display.max_rows = 20
        pandas DataFrame表格最大显示列数
        pd.options.display.max_columns = 20
        pandas精度浮点数显示4位
        pd.options.display.precision = 4
        numpy精度浮点数显示4位，不使用科学计数法
        np.set_printoptions(precision=4, suppress=True)
    """


def encrypt(obj):

    """
        method : md5(), sha1(), sha224(), sha256(), sha384(), and sha512()
        algorithms may be available depending upon the OpenSSL library that Python uses on your platform.
        e.g. : hashlib.sha224("Nobody inspects the spammish repetition").hexdigest()
    """
    import hashlib
    m = hashlib.md5()
    m.update(obj)
    if hex :
        res = m.hexdigest()
    else:
        res = m.digest()
    return res


def extract(_p_dir,file='RomDataBu/df_kl.h5.zip'):
    """
    解压数据
    """
    data_example_zip = os.path.join(_p_dir, file)
    try:
        from zipfile import ZipFile

        zip_h5 = ZipFile(data_example_zip, 'r')
        unzip_dir = os.path.join(_p_dir, 'RomDataBu/')
        for h5 in zip_h5.namelist():
            zip_h5.extract(h5, unzip_dir)
        zip_h5.close()
    except Exception as e:
        print('example env failed! e={}'.format(e))


def verify_indices_all_unique(obj):

    axis_names = [
        ('index',),                            # Series
        ('index', 'columns'),                  # DataFrame
        ('items', 'major_axis', 'minor_axis')  # Panel
    ][obj.ndim - 1]  # ndim = 1 should go to entry 0,

    for axis_name, index in zip(axis_names, obj.axes):
        if index.is_unique:
            continue

        raise ValueError(
            "Duplicate entries in {type}.{axis}: {dupes}.".format(
                type=type(obj).__name__,
                axis=axis_name,
                dupes=sorted(index[index.duplicated()]),
            )
        )
    return obj


def validate_keys(dict_, expected, funcname):
    """Validate that a dictionary has an expected set of keys.
    """
    expected = set(expected)
    received = set(dict_)

    missing = expected - received
    if missing:
        raise ValueError(
            "Missing keys in {}:\n"
            "Expected Keys: {}\n"
            "Received Keys: {}".format(
                funcname,
                sorted(expected),
                sorted(received),
            )
        )

    unexpected = received - expected
    if unexpected:
        raise ValueError(
            "Unexpected keys in {}:\n"
            "Expected Keys: {}\n"
            "Received Keys: {}".format(
                funcname,
                sorted(expected),
                sorted(received),
            )
        )


# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
def cache_dir(environ):
    try:
        return environ['EMPYRICAL_CACHE_DIR']
    except KeyError:
        return join(

            environ.get(
                'XDG_CACHE_HOME',
                expanduser('~/.cache/'),
            ),
            'empyrical',
        )


# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
def data_path(name):
    return os.join(cache_dir(), name)


# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
def ensure_directory(path):
    """
    Ensure that a directory named "path" exists.
    """

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != os.errno.EEXIST or not os.isdir(path):
            raise


def get_utc_timestamp(dt):
    """
    Returns the Timestamp/DatetimeIndex
    with either localized or converted to UTC.

    Parameters
    ----------
    dt : Timestamp/DatetimeIndex
        the date(s) to be converted

    Returns
    -------
    same type as input
        date(s) converted to UTC
    """

    dt = pd.to_datetime(dt)
    try:
        dt = dt.tz_localize('UTC')
    except TypeError:
        dt = dt.tz_convert('UTC')
    return dt


# if not os.path.exists(g_project_kl_df_data_example):
#     # 如果还没有进行解压，开始解压df_kl.h5.zip
#     data_example_zip = os.path.join(_p_dir, 'RomDataBu/df_kl.h5.zip')
#     try:
#         from zipfile import ZipFile
#         zip_h5 = ZipFile(data_example_zip, 'r')
#         unzip_dir = os.path.join(_p_dir, 'RomDataBu/')
#         for h5 in zip_h5.namelist():
#             zip_h5.extract(h5, unzip_dir)
#         zip_h5.close()
#     except Exception as e:
#         # 解压测试数据zip失败，就不开启测试数据模式了
#         print('example env failed! e={}'.format(e))
#     return

def check_indexes_all_same(indexes, message="Indexes are not equal."):
    """Check that a list of Index objects are all equal.

    Parameters
    ----------
    indexes : iterable[pd.Index]
        Iterable of indexes to check.
    message : string

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
            # 返回非0元素的位置
            bad_loc = np.flatnonzero(~same)[0]
            raise ValueError(
                "{}\nFirst difference is at index {}: "
                "{} != {}".format(
                    message, bad_loc, first[bad_loc], other[bad_loc]
                ),
            )


g_project_log_dir = 'c_test'


def init_logging(g_project_log_info):
    """
    logging相关初始化工作，配置log级别，默认写入路径，输出格式
    """
    if not os.path.exists(g_project_log_dir):
        # 创建log文件夹
        os.makedirs(g_project_log_dir)

    # 输出格式规范
    # file_handler = logging.FileHandler(g_project_log_info, 'a', 'utf-8')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=g_project_log_info,
                        filemode='a'
                        # handlers=[file_handler]
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 屏幕打印只显示message
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


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


def is_sorted_ascending(a):
    """Check if a numpy array is sorted."""
    return (np.fmax.accumulate(a) <= a).all()


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
    indices = np.flatnonzero(np.diff(a)) + 1

    if not include_first:
        return indices

    return np.hstack([[0], indices])


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


def compare_datetime_arrays(x, y):
    """
    Compare datetime64 ndarrays, treating NaT values as equal.
    """

    return np.array_equal(x.view('int64'), y.view('int64'))


object_dtype = np.dtype('O')
# We use object arrays for strings.
categorical_dtype = object_dtype


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
