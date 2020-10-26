# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from decimal import Decimal
import math, numpy as np, pandas as pd
from numpy import isnan
from scipy import integrate


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


def demean(row):
    return row - np.nanmean(row)


def zoom(raw):
    if isinstance(raw, (pd.Series, pd.DataFrame)):
        scale = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        raw = np.array(raw)
        scale = (raw - min(raw)) / (max(raw) - min(raw))
    return scale


def standardize(raw):
    standard = (raw - raw.mean()) / raw.std()
    return standard


def zscore(row):
    return (row - np.nanmean(row)) / np.nanstd(row)


# 弧度转角度
def coef2deg(x):
    rad = np.math.acos(x)
    deg = np.rad2deg(rad)
    return deg


# 积分
def funcScorer(func, *args):
    area, err = integrate.quad(func, *args)
    ratio = (area - err) / area
    return area, ratio


def Euclidean(x, y):
    """
        1 /（1 + 距离） y和y_fit的euclidean欧式距离(L2范数)、点与点之间的绝对距离
        扩展：
            1、 y和y_fit的manhattan曼哈顿距离(L1范数) 坐标之间的绝对距离之和
            2、 y和y_fit切比雪夫距离 max(各个维度的最大值)
    """
    x_scale = zoom(x)
    y_scale = zoom(y)
    distance = np.sqrt((x_scale - y_scale) ** 2)
    p = 1 / (1 + distance)
    return p


def CosDistance(x, y):
    """
        1、余弦相似度（夹角，90无关，0为相似度1） y和y_fit的cosine余弦距离（相似性） 向量之间的余弦值
    """
    x_s = zoom(x)
    y_s = zoom(y)
    cos = x.y / (np.sqrt(x_s ** 2) * np.sqrt(y_s ** 2))
    return cos


def CovDistance(x, y):
    """
        1、基于相关系数数学推导， 协方差与方差 马氏距离 将数据投影到N(0, 1)  区间并求其欧式距离，称为数据的协方差距离
    """
    x_s = zoom(x)
    y_s = zoom(y)
    cov = (x * y).mean() - (x_s.mean()) * y_s.mean()
    p = cov / (x.std() * y.std())
    return p


def SignDistance(x, y):

    """
        1、符号＋－相关系数, sign_x = np.sign(x), sign_y = np.sign(y), np.corrcoef(sign_x, sign_y)[0][1]
    """
    sign_x = np.sign(x)
    sign_y = np.sign(y)
    p = (sign_x * sign_y).sum() / len(x)
    return p


def RankDistance(x, y):
    """
        1、排序rank(ascending=False, method='first') 计算相关系数
    """
    x_rank = x.rank()
    y_rank = y.rank()
    p = CovDistance(x_rank, y_rank)
    return p


def _fit_poly(y, degree):
    # return n_array (dimension ascending = False) p(x) = p[0] * x**deg + ... + p[deg]
    if isinstance(y, pd.Series):
        y.dropna(inplace=True)
    res = np.polyfit(range(len(y)), np.array(y), degree)
    return res[0]


def _fit_lstsq(x, y):
    res = np.linalg.lstsq(x, y)
    return res[0][0]


def winsorize(row, min_percentile, max_percentile):
    """
    This implementation is based on scipy.stats.mstats.winsorize
    """
    a = row.copy()
    nan_count = np.isnan(row).sum()
    nonnan_count = a.size - nan_count

    # NOTE: argsort() sorts nans to the end of the array.
    idx = a.argsort()

    # Set values at indices below the min percentile to the value of the entry
    # at the cutoff.
    if min_percentile > 0:
        lower_cutoff = int(min_percentile * nonnan_count)
        a[idx[:lower_cutoff]] = a[idx[lower_cutoff]]

    # Set values at indices above the max percentile to the value of the entry
    # at the cutoff.
    if max_percentile < 1:
        upper_cutoff = int(np.ceil(nonnan_count * max_percentile))
        # if max_percentile is close to 1, then upper_cutoff might not
        # remove any values.
        if upper_cutoff < nonnan_count:
            start_of_nans = (-nan_count) if nan_count else None
            a[idx[upper_cutoff:start_of_nans]] = a[idx[upper_cutoff - 1]]

    return a


def quantiles(data, nbins_or_partition_bounds):
    """
    Compute rowwise array quantiles on an input.
    quartiles -4  quintiles -5  deciles -10

    """
    return np.apply_along_axis(
        np.qcut,
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
    # out=empty_like(data, dtype=self.dtype),
    """
    if out is None:
        out = np.empty_like(data)

    for (row, label_row, out_row) in zip(data, group_labels, out):
        for label in np.unique(label_row):
            locs = (label_row == label)
            out_row[locs] = func(row[locs], *func_args)
    return out
