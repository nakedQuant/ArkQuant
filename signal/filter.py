# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
from numpy import (
    float64,
    nan,
    nanpercentile
)
from functools import partial
import pandas as pd, numpy as np

int64_dtype = np.dtype('int64')
METHOD_OF_CATEGORY = frozenset(['bins', 'quantiles'])


def validate_type_method(raw, kwargs):
    if len(np.unique(raw.dtypes)) - 1:
        raise TypeError('the dtype of raw must be the same')
    if kwargs['method'] in METHOD_OF_CATEGORY and kwargs['bins']:
        assert True, ('expect method in %r,but received %s'
                      % (METHOD_OF_CATEGORY, kwargs['method']))


class Classifier(object):

    missing_value = [None, np.nan, np.NaN, np.NAN]

    def __setattr__(self, key, value):
        raise NotImplementedError

    # 基于integer encode
    def auto_encode(self, raw, **kwargs):
        validate_type_method(raw, kwargs)
        # Return a contiguous flattened array
        non_unique = set(np.ravel(raw)) - set(self.missing_value)
        bin = kwargs['bins']
        encoding = pd.cut(non_unique, bins=bin, labes=range(len(bin)))
        return zip(non_unique, encoding)

    @staticmethod
    # func --- integer
    def custom_encode(data, encoding_function, **kwargs):
        if kwargs:
            encoding_map = partial(encoding_function, **kwargs)
        else:
            encoding_map = encoding_function
        # otypes --- type
        encoding = np.vectorize(encoding_map, otypes=[int64_dtype])(data)
        return zip(data, encoding)


def is_missing(data, missing_value):
    """
    Generic is_missing function that handles NaN and NaT.
    """
    return data == missing_value


def concat_tuples(*tuples):
    """
    Concatenate a sequence of tuples into one tuple.
    """
    return tuple(chain(*tuples))


class Filter(object):
    """
    Pipeline expression computing a boolean output.

    Filters are most commonly useful for describing sets of asset to include
    or exclude for some particular purpose. Many Pipeline API functions accept
    a ``mask`` argument, which can be supplied a Filter indicating that only
    values passing the Filter should be considered when performing the
    requested computation. For example, :meth:`zipline.pipe.Factor.top`
    accepts a mask indicating that ranks should be computed only on asset that
    passed the specified Filter.

    The most common way to construct a Filter is via one of the comparison
    operators (``<``, ``<=``, ``!=``, ``eq``, ``>``, ``>=``) of

    Filters can be combined via the ``&`` (and) and ``|`` (or) operators.

    ``&``-ing together two filters produces a new Filter that produces True if
    **both** of the inputs produced True.

    ``|``-ing together two filters produces a new Filter that produces True if
    **either** of its inputs produced True.

    The ``~`` operator can be used to invert a Filter, swapping all True values
    with Falses and vice-versa.

    Filters may be set as the ``screen`` attribute of a Pipeline, indicating
    asset/date pairs for which the filter produces False should be excluded
    from the Pipeline's output.  This is useful both for reducing noise in the
    output of a Pipeline and for reducing memory consumption of Pipeline
    results.
    """


class NullFilter(Filter):
    """
    A Filter indicating whether input values are missing from an input.

    Parameters
    ----------
    factor : zipline.pipeline.Term
        The factor to compare against its missing_value.
    """
    window_length = 0

    def __new__(cls, missing_value):
        return super(NullFilter, cls).__new__(
            cls,
            missing_value,
        )

    def _compute(self, arrays, dates, assets, mask):
        return is_missing(arrays, self.missing_value)


class NotNullFilter(Filter):
    """
    A Filter indicating whether input values are **not** missing from an input.

    Parameters
    ----------
    factor : zipline.pipeline.Term
        The factor to compare against its missing_value.
    """
    window_length = 0

    def __new__(cls, missing_value):
        return super(NotNullFilter, cls).__new__(
            cls,
            missing_value,
        )

    def _compute(self, arrays, dates, assets, mask):
        return ~is_missing(arrays, self.missing_value)


class PercentileFilter(Filter):
    """
    A Filter representing asset falling between percentile bounds of a Factor.

    Parameters
    ----------
    factor : zipline.pipe.factor.Factor
        The factor over which to compute percentile bounds.
    min_percentile : float [0.0, 1.0]
        The minimum percentile rank of an asset that will pass the filter.
    max_percentile : float [0.0, 1.0]
        The maxiumum percentile rank of an asset that will pass the filter.
    """
    def __new__(cls, min_percentile, max_percentile, mask):
        return super(PercentileFilter, cls).__new__(
            cls,
            mask=mask,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )

    def _validate(self):
        """
        Ensure that our percentile bounds are well-formed.
        """
        if not 0.0 <= self._min_percentile < self._max_percentile <= 100.0:
            raise BadPercentileBounds(
                min_percentile=self._min_percentile,
                max_percentile=self._max_percentile,
                upper_bound=100.0
            )

    def _compute(self, arrays, dates, assets, mask):
        """
        For each row in the input, compute a mask of all values falling between
        the given percentiles.
        """
        data = arrays.copy().astype(float64)
        data[~mask] = nan

        lower_bounds = nanpercentile(
            data,
            self._min_percentile,
            axis=1,
            keepdims=True,
        )
        upper_bounds = nanpercentile(
            data,
            self._max_percentile,
            axis=1,
            keepdims=True,
        )
        return (lower_bounds <= data) & (data <= upper_bounds)


class ArrayPredicate(Filter):
    """
    A filter applying a function from (ndarray, *args) -> ndarray[bool].

    Parameters
    ----------
    term : zipline.pipeline.Term
        Term producing the array over which the predicate will be computed.
    op : function(ndarray, *args) -> ndarray[bool]
        Function to apply to the result of `term`.
    opargs : tuple[hashable]
        Additional argument to apply to ``op``.
    """
    params = ('op', 'opargs')
    window_length = 0

    def __new__(cls, op, opargs):
        hash(opargs)  # fail fast if opargs isn't hashable.
        return super(ArrayPredicate, cls).__new__(
            ArrayPredicate,
            op=op,
            opargs=opargs,
        )

    def _compute(self, arrays, dates, assets, mask):
        params = self.params
        return params['op'](arrays, *params['opargs']) & mask


class BusinessDaysEvent(object):
    """
    Abstract class for business days since a previous event.
    Returns the number of **business days** (not trading days!) since
    the most recent event date for each asset.

    This doesn't use trading days for symmetry with
    BusinessDaysUntilNextEarnings.

    asset which announced or will announce the event today will produce a
    value of 0.0. asset that announced the event on the previous business
    day will produce a value of 1.0.

    asset for which the event date is `NaT` will produce a value of `NaN`.

    Factors describing information about event data (e.g. earnings
    announcements, acquisitions, dividends, etc.).

    Dividend.com is a financial services website focused on providing comprehensive dividend stock research information.
    The company uses their proprietary DARS™, or Dividend Advantage Rating System, to rank nearly 1,600 dividend-paying
    stocks across five distinct criteria: relative strength, overall yield attractiveness, dividend reliability,
    dividend uptrend, and earnings growth.

    Automatic Trendline Detection
    Support and Resistance Visualizations
    Automatic Fibonacci Retracements
    Manual Trendline Tuning Mode
    Automated Candlestick Pattern Detection
    Automated Price Gap Detection

    trade via volatity

    波动率 --- 并非常数，均值回复，聚集，存在长期记忆性的
    大收益率会发生的相对频繁 --- 存在后续的波动
    在大多数市场中，波动率与收益率呈现负相关，在股票市场中的最为明显
    波动率和成交量之间存在很强的正相关性
    波动率分布接近正太分布

    基于大类过滤
    标的 --- 按照市值排明top 10% 标的标的集合 --- 度量周期的联动性
    a 计算每个月的月度收益率，筛选出10%集合 / 12的个数，
    b 获取每个月的集合 --- 作为当月的强周期集合
    c 基于技术指标等技术获取对应的标的

    1、序列中基于中位数的性质更加能代表趋势动向
    2、预警指标的出现股票集中，容易出现后期的大黑马，由此推导出异动逻辑以及在持续性
    3、统计套利:(pt > pt-1) / (pt-1 > m) 概率为75% 引发的思考(close - pre_high)

    """
    # numpy.busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None)
    # Counts the number of valid days between begindates and enddates, not including the day of enddates.
    #  [1,1,1,1,1,0,0]; a length-seven string, like ‘1111100’; or a string like “Mon Tue Wed Thu Fri Sat Sun”
    # np.busday_count --- Mon Fri ( weekmaskstr or array_like of bool, optional)
    # where(condition, x,y --- handles `NaT`s by returning `NaN`s where the inputs were `NaT`.
    # announce_date , ex_date , pay_date (比如举牌、增持、减持、股权转让、重组）


