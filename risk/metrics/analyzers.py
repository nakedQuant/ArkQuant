#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd, numpy as np
from scipy import stats

__all__ = [
    'cum_returns_final',
    'annual_return',
    'annual_volatility',
    'sharpe_ratio',
    'sqn',
    'excess_sharpe',
    'calmar_ratio',
    'stability_of_timeseries',
    'max_drawdown',
    'omega_ratio',
    'sortino_ratio',
    # 'stats.skew',
    # 'stats.kurtosis',
    'tail_ratio',
    'cagr',
    'value_at_risk',
    'conditional_value_at_risk',
]


def cum_returns(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """

    if len(returns) < 1:
        return returns.copy()

    out = np.empty_like(returns)

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)
    # cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())
    np.subtract(out, 1, out=out)
    # np.multiply(out, starting_value, out=out)
    if returns.ndim == 1 and isinstance(returns, pd.Series):
        out = pd.Series(out, index=returns.index)
    elif isinstance(returns, pd.DataFrame):
        out = pd.DataFrame(
            out, index=returns.index, columns=returns.columns,
            )
    return out


def cum_returns_final(
                    returns,
                    risk_free=0.0,
                    required_return=0.0
                    ):
    """
    Compute total returns from simple returns.

    Parameters
    ----------
    returns : pd.DataFrame, pd.Series, or np.ndarray
       Noncumulative simple returns of one or more timeseries.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    total_returns : pd.Series, np.ndarray, or float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a
        scalar.

        If input is 2-dimensional (a DataFrame or 2D numpy array), the result
        is a 1D array containing cumulative returns for each column of input.
    """
    if len(returns) == 0:
        return np.nan
    if isinstance(returns, pd.DataFrame):
        result = (returns + 1).prod()
    else:
        result = np.nanprod(returns + 1, axis=0)
    return result


def max_drawdown(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    drawdown, drawdown_max, duration
    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    if len(returns) < 1:
        return np.nan
    out = np.empty(returns.shape[1:])
    returns_array = np.asanyarray(returns)
    cumulative = cum_returns(returns_array)
    max_return = np.fmax.accumulate(cumulative, axis=0)
    np.nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    # 久期 duration --- 1.整个下跌的时间 ， 2.连续性
    #计算阴跌的最长期限
    mask = pd.Series(np.where(out == 0 , 0, 1))
    duration = np.sum(mask)
    # 返回1对应的idx
    mask_diff = list(np.diff(mask[mask == 1].index))
    # 追加0  --- length
    mask_diff.insert(0,0)
    mask_diff.append(len(mask_diff))
    # 替换 非1
    sub_mask = pd.Series(np.where(mask_diff != 1,0,1))
    sub_mask_loc = list(sub_mask[sub_mask == 0].index)
    succession = max(np.diff(sub_mask_loc))
    return out ,duration , succession


def annual_return(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the mean annual growth rate of returns. This is equivilent
    to the compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
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

    Returns
    -------
    annual_return : float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """
    if len(returns) < 1:
        return np.nan

    num_years = len(returns) / 252
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns)
    return ending_value ** (1 / num_years) - 1


def annual_volatility(
                    returns,
                    risk_free=0.0,
                    required_return=0.0
                    ):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    'monthly':12
    'weekly': 52
    'daily': 252

    Returns
    -------
    annual_volatility : float
    """
    if len(returns) < 2:
        return  np.nan
    out = np.empty(returns.shape[1:])
    np.nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, 252 ** (1.0 / 2), out=out)
    return out


def cagr(
        returns,
        risk_free=0.0,
        required_return=0.0
        ):
    """
    Compute compound annual growth rate. Alias function for
    :func:`~empyrical.stats.annual_return`

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free :  treasure interest
    required_return : benchmark return

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

    Returns
    -------
    cagr : float
        The CAGR value.

    """
    return annual_return(returns)


def calmar_ratio(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
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


    Returns
    -------
    calmar_ratio : float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd , duration , succession = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(returns=returns) / abs(max_dd)
    else:
        return np.nan
    if np.isinf(temp):
        return np.nan
    return temp


def omega_ratio(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period
    required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.
    annualization : int, optional
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.

    Returns
    -------
    omega_ratio : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """
    if len(returns) < 2:
        return np.nan
    if required_return <= -1:
        return np.nan
    else:
        # return_threshold = required_return
        return_threshold = (1 + required_return) ** \
            (1. / 252) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant daily risk-free return throughout the period.
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
    sharpe_ratio : float
        nan if insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """
    if len(returns) < 2:
        return np.nan

    # return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)
    out = np.empty(returns.shape[1:])
    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    np.multiply(
        np.divide(
            np.nanmean(returns_risk_adj, axis=0),
            np.nanstd(returns_risk_adj, ddof=1, axis=0),
            out=out,
        ),
        # period --- daily
        np.sqrt(252),
        out=out,
    )
    return out


def excess_sharpe(
                excess_returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the Excess Sharpe of a strategy.

    Parameters
    ----------
    excess_returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
        returns - factor_returns : float / series Benchmark return to compare returns against.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    excess_sharpe : float

    Note
    -----
    The excess Sharpe is a simplified Information Ratio that uses
    tracking error rather than "active risk" as the denominator.
    """
    # if len(returns) < 2:
    #     return np.nan
    # out = np.empty(returns.shape[1:])
    # active_return = _adjust_returns(returns, factor_returns)
    if len(excess_returns) < 2:
        return np.nan
    out = np.empty(excess_returns.shape[1:])
    active_return = excess_returns
    tracking_error = np.nan_to_num(np.nanstd(active_return, ddof=1, axis=0))

    out = np.divide(
        np.nanmean(active_return, axis=0, out=out),
        tracking_error,
        out=out,
    )
    return out


def sortino_ratio(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
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
    _downside_risk : float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sortino_ratio : float or pd.Series

        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details.

    """
    # return np.sqrt(periods) * (np.mean(returns)) / np.std(returns[returns < 0])
    if len(returns) < 2:
        return np.nan

    out = np.empty(returns.shape[1:])

    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))
    # period=DAILY
    average_annual_return = np.nanmean(adj_returns, axis=0) * 252
    annualized_downside_risk = downside_risk(returns, required_return)

    np.divide(average_annual_return, annualized_downside_risk, out=out)
    if isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


def downside_risk(
                returns,
                risk_free=0.0,
                required_return=0
                ):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
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
    downside_deviation : float or pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details, specifically why using the
    standard deviation of the negative returns is not correct.
    """
    if len(returns) < 1:
        return np.nan

    out = np.empty(returns.shape[1:])

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        np.NINF,
        0,
    )
    np.square(downside_diff, out=downside_diff)
    np.nanmean(downside_diff, axis=0, out=out)
    np.sqrt(out, out=out)
    # period = 'daily'
    np.multiply(out, np.sqrt(252), out=out)
    if isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)
    return out


def sqn(
        returns,
        risk_free=0.0,
        required_return=0.0
        ):
    """
    SQN or SystemQualityNumber. Defined by Van K. Tharp to categorize trading
    systems.

      - 1.6 - 1.9 Below average
      - 2.0 - 2.4 Average
      - 2.5 - 2.9 Good
      - 3.0 - 5.0 Excellent
      - 5.1 - 6.9 Superb
      - 7.0 -     Holy Grail?

    The formula:

      - SquareRoot(NumberTrades) * Average(TradesProfit) / StdDev(TradesProfit)

    The sqn value should be deemed reliable when the number of trades >= 30

    Methods:

      - get_analysis

        Returns a dictionary with keys "sqn" and "trades" (number of
        considered trades)
    """
    sqn = np.sqrt(len(returns)) * np.nanmean(returns) / np.nanstd(returns)
    return sqn


def stability_of_timeseries(
                            returns,
                            risk_free=0.0,
                            required_return=0.0
                            ):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns)[2]

    return rhat ** 2


def tail_ratio(
                returns,
                risk_free=0.0,
                required_return=0.0
                ):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    tail_ratio : float
    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def value_at_risk(
                returns,
                risk_free=0.0,
                required_return=0.0,
                cutoff=0.05
                ):
    """
    Value at risk (VaR) of a returns stream.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.

    Returns
    -------
    VaR : float
        The VaR value.
    """
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(
                            returns,
                            risk_free=0.0,
                            required_return=0.0,
                            cutoff=0.05
                            ):
    """
    Conditional value at risk (CVaR) of a returns stream.

    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.

    Returns
    -------
    CVaR : float
        The CVaR value.
    """
    # PERF: Instead of using the 'value_at_risk' function to find the cutoff
    # value, which requires a call to numpy.percentile, determine the cutoff
    # index manually and partition out the lowest returns values. The value at
    # the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)
    # 将小于cutoff_index 值均值
    # partition --- Creates a copy of the array with its elements rearranged in such a way that the value of the element
    # in k-th position is in the position it would be in a sorted array. All elements smaller than the k-th element are
    # moved before this element and all equal or greater are moved behind it. The ordering of the elements in the two
    # partitions is undefined.
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])
