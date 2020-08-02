#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd , numpy as np
from scipy.stats import linregress
from collections import OrderedDict
from .uility import _to_pandas , annualization_factor , _adjust_returns ,roll
from .analyzers import  annual_return


def rsquared(x, y):
    """
    Return R^2 where x and y are array-like.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2


def _aligned_series(*many_series):
    """
    Return a new list of series containing the data in the input series, but
    with their indices aligned. NaNs will be filled in for missing values.

    Parameters
    ----------
    *many_series
        The series to align.

    Returns
    -------
    aligned_series : iterable[array-like]
        A new list of series containing the data in the input series, but
        with their indices aligned. NaNs will be filled in for missing values.

    """
    head = many_series[0]
    tail = many_series[1:]
    n = len(head)
    if (isinstance(head, np.ndarray) and
            all(len(s) == n and isinstance(s, np.ndarray) for s in tail)):
        # optimization: ndarrays of the same length are already aligned
        return many_series

    # 如果不一致 -- 拼接 --- 拆分
    return (
        v
        for _, v in (pd.concat(map(_to_pandas, many_series), axis=1)).values()
    )


def alpha_aligned(returns,
                  factor_returns,
                  risk_free=0.0,
                  period='daily',
                  annualization=None,
                  out=None,
                  _beta=None):
    """Calculates annualized alpha.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:], dtype='float64')

    if len(returns) < 2:
        out[()] = np.nan
        if returns.ndim == 1:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        _beta = beta_aligned(returns, factor_returns, risk_free)

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    out = np.subtract(
        np.power(
            np.add(
                np.nanmean(alpha_series, axis=0, out=out),
                1,
                out=out
            ),
            ann_factor,
            out=out
        ),
        1,
        out=out
    )

    if allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    if returns.ndim == 1:
        out = out.item()

    return out


def alpha(returns,
          factor_returns,
          risk_free=0.0,
          period='daily',
          annualization=None,
          out=None,
          _beta=None):
    """Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~empyrical.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    float
        Alpha.
    """
    if not (isinstance(returns, np.ndarray) and
            isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
        _beta=_beta
    )

def beta_aligned(returns, factor_returns, risk_free=0.0, out=None):
    """Calculates beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta : float
        Beta.
    """
    # Cache these as locals since we're going to call them multiple times.
    nan = np.nan
    isnan = np.isnan

    returns_1d = returns.ndim == 1
    if returns_1d:
        returns = returns[:, np.newaxis]

    if factor_returns.ndim == 1:
        factor_returns = factor_returns[:, np.newaxis]

    N, M = returns.shape

    if out is None:
        out = np.full(M, nan)
    elif out.ndim == 0:
        out = out[np.newaxis]

    if len(returns) < 1 or len(factor_returns) < 2:
        out[()] = nan
        if returns_1d:
            out = out.item()
        return out

    # Copy N times as a column vector and fill with nans to have the same
    # missing value pattern as the dependent variable.
    # PERF_TODO: We could probably avoid the space blowup by doing this in
    # Cython.
    # shape: (N, M)
    independent = np.where(
        isnan(returns),
        nan,
        factor_returns,
    )

    # 计算序列协方差的公式
    ind_residual = independent - np.nanmean(independent, axis=0)
    # mean((X - mean(X)) * (Y - mean(Y))) --- X_res = (X - mean(X)) ， mean(X_res * Y)
    covariances = np.nanmean(ind_residual * returns, axis=0)

    # We end up with different variances in each column here because each
    # column may have a different subset of the data dropped due to missing
    # data in the corresponding dependent column.
    # shape: (M,)
    np.square(ind_residual, out=ind_residual)
    independent_variances = np.nanmean(ind_residual, axis=0)
    independent_variances[independent_variances < 1.0e-30] = np.nan
    # Calculate beta as Cov(X, Y) / Cov(X, X).
    np.divide(covariances, independent_variances, out=out)

    if returns_1d:
        out = out.item()

    return out


def beta(returns, factor_returns, risk_free=0.0, out=None):
    """Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta : float
    """
    if not (isinstance(returns, np.ndarray) and
            isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        out=out,
    )


def alpha_beta_aligned(returns,
                       factor_returns,
                       risk_free=0.0,
                       period='daily',
                       annualization=None,
                       out=None):
    """Calculates annualized alpha and beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    beta : float
    """
    if out is None:
        out = np.empty(returns.shape[1:] + (2,), dtype='float64')

    b = beta_aligned(returns, factor_returns, risk_free, out=out[..., 1])
    alpha_aligned(
        returns,
        factor_returns,
        risk_free,
        period,
        annualization,
        out=out[..., 0],
        _beta=b,
    )

    return out

def alpha_beta(returns,
               factor_returns,
               risk_free=0.0,
               period='daily',
               annualization=None,
               out=None):
    """Calculates annualized alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    beta : float
    """
    returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
    )

# def roll_alpha_beta(returns, factor_returns, window=10, **kwargs):
#     """
#     Computes alpha and beta over a rolling window.
#
#     Parameters
#     ----------
#     lhs : array-like
#         The first array to pass to the rolling alpha-beta.
#     rhs : array-like
#         The second array to pass to the rolling alpha-beta.
#     window : int
#         Size of the rolling window in terms of the periodicity of the data.
#     out : array-like, optional
#         Array to use as output buffer.
#         If not passed, a new array will be created.
#     **kwargs
#         Forwarded to :func:`~empyrical.alpha_beta`.
#     """
#     returns, factor_returns = _aligned_series(returns, factor_returns)
#
#     return roll_alpha_beta_aligned(
#         returns,
#         factor_returns,
#         window=window,
#         **kwargs
#     )

def beta_fragility_heuristic(returns, factor_returns):
    """Estimate fragility to drops in beta.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    Note
    ----
    A negative return value indicates potential losses
    could follow volatility in beta.
    The magnitude of the negative value indicates the size of
    the potential loss.
    seealso::
    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    return beta_fragility_heuristic_aligned(
        *_aligned_series(returns, factor_returns))


def beta_fragility_heuristic_aligned(returns, factor_returns):
    """Estimate fragility to drops in beta

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    Note
    ----
    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.
    seealso::
    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    # combine returns and factor returns into pairs
    returns_series = pd.Series(returns)
    factor_returns_series = pd.Series(factor_returns)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ['returns', 'factor_returns']

    # exclude any rows where returns are nan
    pairs = pairs.dropna()
    # sort by beta
    pairs = pairs.sort_values(by='factor_returns')

    # find the three vectors, using median of 3
    start_index = 0
    mid_index = int(np.around(len(pairs) / 2, 0))
    end_index = len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]

    factor_returns_range = (end_factor_returns - start_factor_returns)
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    # find weights for the start and end returns
    # using a convex combination
    if not factor_returns_range == 0:
        start_returns_weight = \
            (mid_factor_returns - start_factor_returns) / \
            factor_returns_range
        end_returns_weight = \
            (end_factor_returns - mid_factor_returns) / \
            factor_returns_range

    # calculate fragility heuristic
    heuristic = (start_returns_weight*start_returns) + \
        (end_returns_weight*end_returns) - mid_returns

    return heuristic

def perf_attrib(returns,
                positions,
                factor_returns,
                factor_loadings):
    """
    Attributes the performance of a returns stream to a set of risk factors.

    Performance attribution determines how much each risk factor, e.g.,
    momentum, the technology sector, etc., contributed to total returns, as
    well as the daily exposure to each of the risk factors. The returns that
    can be attributed to one of the given risk factors are the
    `common_returns`, and the returns that _cannot_ be attributed to a risk
    factor are the `specific_returns`. The `common_returns` and
    `specific_returns` summed together will always equal the total returns.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.
        - Example:
            2017-01-01   -0.017098
            2017-01-02    0.002683
            2017-01-03   -0.008669

    positions: pd.Series
        Daily holdings in percentages, indexed by date.
        - Examples:
            dt          ticker
            2017-01-01  AAPL      0.417582
                        TLT       0.010989
                        XOM       0.571429
            2017-01-02  AAPL      0.202381
                        TLT       0.535714
                        XOM       0.261905

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns
        - Example:
                        momentum  reversal
            2017-01-01  0.002779 -0.005453
            2017-01-02  0.001096  0.010290

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    Returns
    -------
    tuple of (risk_exposures_portfolio, perf_attribution)

    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    perf_attribution : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980

    Note
    ----
    See https://en.wikipedia.org/wiki/Performance_attribution for more details.
    """

    # Make risk data match time range of returns
    start = returns.index[0]
    end = returns.index[-1]
    factor_returns = factor_returns.loc[start:end]
    factor_loadings = factor_loadings.loc[start:end]

    factor_loadings.index = factor_loadings.index.set_names(['dt', 'ticker'])

    positions = positions.copy()
    positions.index = positions.index.set_names(['dt', 'ticker'])

    risk_exposures_portfolio = compute_exposures(positions,
                                                 factor_loadings)

    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
    common_returns = perf_attrib_by_factor.sum(axis='columns')

    tilt_exposure = risk_exposures_portfolio.mean()
    tilt_returns = factor_returns.multiply(tilt_exposure).sum(axis='columns')
    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame(OrderedDict([
        ('total_returns', returns),
        ('common_returns', common_returns),
        ('specific_returns', specific_returns),
        ('tilt_returns', tilt_returns),
        ('timing_returns', timing_returns)
        ]))

    return (risk_exposures_portfolio,
            pd.concat([perf_attrib_by_factor, returns_df], axis='columns'))


def compute_exposures(positions, factor_loadings):
    """
    Compute daily risk factor exposures.

    Parameters
    ----------
    positions: pd.Series
        A series of holdings as percentages indexed by date and ticker.
        - Examples:
            dt          ticker
            2017-01-01  AAPL      0.417582
                        TLT       0.010989
                        XOM       0.571429
            2017-01-02  AAPL      0.202381
                        TLT       0.535714
                        XOM       0.261905

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    Returns
    -------
    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515
    """
    risk_exposures = factor_loadings.multiply(positions, axis='rows')
    return risk_exposures.groupby(level='dt').sum()


def up(returns, factor_returns, **kwargs):
    """
    Calculates a given statistic filtering only positive factor return periods.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns (optional): float / series
        Benchmark return to compare returns against.
    function:
        the function to run for each rolling window.
    (other keywords): other keywords that are required to be passed to the
        function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the function
    """
    func = kwargs.pop('function')
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    return func(returns, factor_returns, **kwargs)


def down(returns, factor_returns, **kwargs):
    """
    Calculates a given statistic filtering only negative factor return periods.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns (optional): float / series
        Benchmark return to compare returns against.
    function:
        the function to run for each rolling window.
    (other keywords): other keywords that are required to be passed to the
        function in the 'function' argument may also be passed in.

    Returns
    -------
    Same as the return of the 'function'
    """
    func = kwargs.pop('function')
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    return func(returns, factor_returns, **kwargs)

def capture(returns, factor_returns, period='daily'):
    """Compute capture ratio.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    capture_ratio : float

    Note
    ----
    See http://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    details.
    """
    return (annual_return(returns, period=period) /
            annual_return(factor_returns, period=period))

def up_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is positive

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_capture : float

    Note
    ----
    See http://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    more information.
    """
    return up(returns, factor_returns, function=capture, **kwargs)


def down_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is negative

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    down_capture : float

    Note
    ----
    See http://www.investopedia.com/terms/d/down-market-capture-ratio.asp for
    more information.
    """
    return down(returns, factor_returns, function=capture, **kwargs)


def up_down_capture(returns, factor_returns, **kwargs):
    """
    Computes the ratio of up_capture to down_capture.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_down_capture : float
        the updown capture ratio
    """
    return (up_capture(returns, factor_returns, **kwargs) /
            down_capture(returns, factor_returns, **kwargs))


def up_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is positive.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """
    return up(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def down_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is negative.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    alpha : float
    beta : float
    """
    return down(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def roll_up_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up capture measure over a rolling window.
    see documentation for :func:`~empyrical.stats.up_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window, function=up_capture,
                **kwargs)


def roll_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the down capture measure over a rolling window.
    see documentation for :func:`~empyrical.stats.down_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window, function=down_capture,
                **kwargs)


def roll_up_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up/down capture measure over a rolling window.
    see documentation for :func:`~empyrical.stats.up_down_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window,
                function=up_down_capture, **kwargs)


FACTOR_STAT_FUNCS = [
    alpha,
    beta,
    beta_fragility_heuristic,
    capture,
    up_capture,
    down_capture
]
