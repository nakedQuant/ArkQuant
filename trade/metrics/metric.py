#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import numpy as np
from toolz import groupby
from gateWay.driver.benchmark import get_benchmark_returns


class _ConstantCumulativeRiskMetric(object):
    """A metrics which does not change, ever.

    Notes
    -----
    This exists to maintain the existing structure of the perf packets. We
    should kill this as soon as possible.
    """
    def __init__(self, field, value):
        self._field = field
        self._value = value

    def end_of_session(self, packet, *args):
        packet['cumulative_risk_metrics'][self._field] = self._value

    def end_of_simulation(self,packet,*args):
        packet['cumulative_risk_metrics'][self._field] = self._value


class PNL(object):
    """Tracks daily and cumulative PNL.
    """
    def __init__(self):
        self._previous_pnl = 0.0

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        self._previous_pnl = 0.0

    def start_of_session(self, ledger):
        self._previous_pnl = ledger.portfolio.pnl

    def _end_of_period(self, field, packet, ledger):
        pnl = ledger.portfolio.pnl
        packet[field]['pnl'] = pnl - self._previous_pnl
        packet['cumulative_perf']['pnl'] = pnl

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        self._end_of_period('daily_perf', packet, ledger)


class CashFlow(object):
    """Tracks daily and cumulative cash flow.
    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """

    def __init__(self):
        self._previous_cash_flow = 0.0

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        self._previous_cash_flow = 0.0

    def start_of_session(self, ledger):
        self._previous_cash_flow = ledger.portfolio.cash_flow

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        cash_flow = ledger.portfolio.cash_flow
        packet['daily_perf']['capital_used'] = (
                cash_flow - self._previous_cash_flow
        )
        packet['cumulative_perf']['capital_used'] = cash_flow
        self._previous_cash_flow = cash_flow


class Returns(object):
    """Tracks the daily and cumulative returns of the algorithm.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['returns'] = ledger.daily_returns
        packet['cumulative_perf']['returns'] = ledger.portfolio.returns


class Profits(object):
    """
        track the daily profit of the algorithm
        dict --- asset : profit
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['profit'] = ledger.daily_position_stats(session_ix)

# class MaxLeverage(object):
#     """Tracks the maximum account leverage.
#     """
#     def start_of_simulation(self,
#                             ledger,
#                             benchmark,
#                             sessions):
#         self._max_leverage = 0.0
#
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         self._max_leverage = max(self._max_leverage, ledger.account.leverage)
#         packet['cumulative_risk_metrics']['max_leverage'] = self._max_leverage

class Weights(object):

    def end_of_session(self,
                        packet,
                        ledger,
                        session_ix):
        weights  = ledger.portolio.current_portfolio_weights
        packet['cumulative_risk_metrics']['portfolio_weights'] = weights


class Uility(object):
    """Tracks the capital usage
    """
    def __init__(self):
        self._capital_usage = 0.0

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        self._capital_usage = 0.0

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['cumulative_risk_metrics']['capital_usage'] = ledger.portfolio.uility


class Cushion(object):
    """Tracks the cushion of account
    """
    def __init__(self):
        self.cushion = 1.0

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        self.cushion = 1.0

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['cumulative_risk_metrics']['cushion'] = 1 - ledger.portfolio.uility


class Proportion(object):
    """
        计算持仓按照资产类别计算比例
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        # 按照资产类别计算持仓
        portfolio = ledger.portfolio
        portfolio_position_values = portfolio.portfolio_value - portfolio.start_cash
        # 持仓分类
        protocols = ledger.get_positions()
        mappings = groupby(lambda x : x.asset.asset_type,protocols)
        # 计算大类权重
        from toolz import valmap
        mappings_value = valmap(lambda x : sum([p.amount * p.last_sync_price for p in x]),mappings)
        ratio = valmap(lambda x : x / portfolio_position_values , mappings_value)
        packet['cumulative_risk_metrics']['proportion'] = ratio


class Hitrate(object):
    """
        1、度量算法触发的概率（生成transaction)
        2、算法的胜率（产生正的收益概率）--- 当仓位完全退出时
    """
    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions):
        closed_positions = ledger.position_tracker.record_closed_position
        groups = groupby(lambda p : np.sign(p.last_sync_price - p.cost_basis),closed_positions)
        packet['cumulative_risk_metrics']['hitRate'] = len(groups[1.0]) / len(closed_positions)
        packet['cumulative_risk_metrics']['evenRate'] = len(groups[0.0]) / len(closed_positions)
        packet['cumulative_risk_metrics']['lossRate'] = 1 - len(groups[-1.0]) / len(closed_positions)


class Positions(object):
    """Tracks and analyse daily positions
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['positions'] = ledger.positions()


class Transactions(object):
    """Tracks daily transactions.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['transactions'] = ledger.get_transactions(session_ix)


class PeriodLabel(object):
    """Backwards compat, please kill me.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['cumulative_risk_metrics']['period_label'] = session_ix.strftime('%Y-%m')


class NumTradingDays(object):
    """Report the number of trading days.
    """
    def __init__(self):
        self._num_trading_days = 0

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        self._num_trading_days = 0

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        self._num_trading_days += 1

    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions):
        packet['cumulative_risk_metrics']['trading_days'] = \
            self._num_trading_days


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """
    def __init__(self):
        self.return_series = None

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        #计算基准收益率
        return_series = get_benchmark_returns(benchmark)
        self.return_series = return_series[sessions]

    def end_of_session(self,
                        packet,
                        ledger,
                        session_ix):
        return_series = self.returns_series
        daily_returns_series = return_series[return_series.index <= session_ix]
        # Series.expanding(self, min_periods=1, center=False, axis=0)
        cumulative_annual_volatility = (
            daily_returns_series.expanding(2).std(ddof=1) * np.sqrt(252)
        ).values[-1]
        cumulative_return = np.cumprod(np.array(1+ daily_returns_series)) -1
        packet['daily_perf']['benchmark_return'] = daily_returns_series[-1]
        packet['cumulative_perf']['benchmark_return'] = cumulative_return
        packet['cumulative_perf']['benchmark_annual_volatility'] = cumulative_annual_volatility


class AlphaBeta(object):
    """End of simulation alpha and beta to the benchmark.
    """
    def __init__(self):
        self.return_series  = None

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        return_series = get_benchmark_returns(benchmark)
        self.return_series = return_series[sessions]

    def end_of_simulation(self,
                   packet,
                   ledger,
                   sessions):
        risk = packet['cumulative_risk_metrics']
        benchmark_returns = self.returns_series
        daily_return_series = ledger.daily_returns_series
        alpha, beta = alpha_beta_aligned(
            daily_return_series,
            benchmark_returns)
        if np.isnan(alpha):
            alpha = None
        if np.isnan(beta):
            beta = None

        risk['alpha'] = alpha
        risk['beta'] = beta


class ReturnsStatistic(object):
    """A metrics that reports an end of simulation scalar or time series
    computed from the algorithm returns.

    Parameters
    ----------
    function : callable
        The function to call on the daily returns.
    field_name : str, optional
        The name of the field. If not provided, it will be
        ``function.__name__``.
    e.g.:
        SIMPLE_STAT_FUNCS = [
        cum_returns_final,
        annual_return,
        annual_volatility,
        sharpe_ratio,
        excess_sharpe,
        sqn,
        calmar_ratio,
        stability_of_timeseries,
        max_drawdown,
        omega_ratio,
        sortino_ratio,
        stats.skew,
        stats.kurtosis,
        tail_ratio,
        cagr,
        value_at_risk,
        conditional_value_at_risk,
        ]
    """
    def __init__(self,
                 function,
                 risk_free = 0.0,
                 required_return = 0.0,
                 field_name=None):
        if field_name is None:
            field_name = function.__name__

        self._function = function
        self.return_series = None
        self.risk_free = risk_free
        self.required_return = required_return
        self._field_name = field_name

    def start_of_simulation(self,
                            ledger,
                            benchmark,
                            sessions):
        #计算基准收益率
        return_series = get_benchmark_returns(benchmark)
        self.return_series = return_series[sessions]

    def end_of_session(self,
                        packet,
                        ledger,
                        session_ix
                        ):
        daily_returns_series = ledger.daily_returns_series
        return_series = self.returns_series
        #
        daily_returns = daily_returns_series[daily_returns_series.index <= session_ix]
        benchmark_returns = return_series[return_series.index <= session_ix]
        #主体计算
        res = self._function(
            daily_returns,
            benchmark_returns,
            self.risk_free,
            self.required_return
        )
        if not np.isfinite(res):
            res = None
        packet['cumulative_risk_metrics'][self._field_name] = res