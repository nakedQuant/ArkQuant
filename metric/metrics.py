#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import numpy as np, operator as op
from toolz import groupby, valmap
from metric.exposure import alpha_beta_aligned


class SessionField(object):
    """Like :class:`~zipline.finance.metric.metric.SimpleLedgerField` but
    also puts the current value in the ``cumulative_perf`` section.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['period_close'] = session_ix


class NumTradingDays(object):
    """Report the number of trading days.
    """
    def __init__(self):
        self._num_trading_days = 0

    def start_of_simulation(self,
                            ledger,
                            benchmark_returns,
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


class _ConstantCumulativeRiskMetric(object):
    """A metric which does not change, ever.

       record initial capital
    Notes
    -----
    This exists to maintain the existing structure of the perf packets. We
    should kill this as soon as possible.
    """
    def __init__(self, field, constant=None):
        self._field = field
        self._constant = constant

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['cumulative_risk_metrics'][self._field] = self._constant

    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions):
        packet['cumulative_risk_metrics'][self._field] = self._constant


class DailyLedgerField(object):
    """Like :class:`~zipline.finance.metric.metric.SimpleLedgerField` but
    also puts the current value in the ``cumulative_perf`` section.
        record daily field

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, _packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if _packet_field:
            self._packet_field = _packet_field
        else:
            self._packet_field = ledger_field.rsplit('.')[-1]

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        field = self._packet_field
        packet['daily_perf'][field] = self._get_ledger_field(ledger)


class Transactions(object):
    """Tracks daily transactions.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        packet['daily_perf']['transactions'] = ledger.get_transactions(session_ix)


class PNL(object):
    """Tracks daily and cumulative PNL --- profit
    """
    def __init__(self):
        self._initial_pnl = 0.0

    def _end_of_period(self, field, packet, ledger):
        # pnl --- daily pnl
        pnl = ledger.portfolio.pnl
        self._initial_pnl = self._initial_pnl + pnl
        packet['cumulative_perf']['pnl'] = self._initial_pnl
        packet[field]['pnl'] = pnl

    def end_of_session(self, packet, ledger, session_ix):
        self._end_of_period('daily_perf', packet, ledger)


class Returns(object):
    """Tracks the daily and cumulative returns of the algorithm.
    """
    def __init__(self):
        self._previous_return = 0.0

    def start_of_session(self, ledger):
        self._previous_return = ledger.portfolio.returns

    def _end_of_period(self, field, packet, ledger):
        returns = ledger.portfolio.returns
        packet['cumulative_perf']['returns'] = returns
        packet[field]['returns'] = (returns + 1) / (self._previous_return + 1) - 1

    def end_of_session(self, packet, ledger, session_ix):
        self._end_of_period('daily_perf', packet, ledger)


class HitRate(object):
    """
        1、度量算法触发的概率（生成transaction)
        2、算法的胜率（产生正的收益概率）--- 当仓位完全退出时
    """
    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions):
        # mappings dt : closed_position
        closed = ledger.position_tracker.record_closed_position
        # print('closed', closed)
        closed_positions = groupby(lambda x: x.name, closed.values())
        win_rate = valmap(lambda x: sum([ele for ele in x if x.cost_basis > 0]) / len(x), closed_positions)
        # win_rate --- pipeline_name : hit_rate
        packet['cumulative_risk_metrics']['hitRate'] = win_rate


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """
    def __init__(self):
        self.return_series = None

    def start_of_simulation(self,
                            ledger,
                            benchmark_returns,
                            sessions):
        # 计算基准收益率
        print('benchmark_returns', benchmark_returns)
        self.return_series = benchmark_returns[sessions]

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix):
        return_series = self.return_series
        daily_returns_series = return_series[return_series.index <= session_ix.strftime('%Y-%m-%d')]
        # Series.expanding(self, min_periods=1, center=False, axis=0)
        cumulative_annual_volatility = (
            daily_returns_series.expanding(2).std(ddof=1) * np.sqrt(252)
        ).values[-1]
        cumulative_return = np.cumprod(np.array(1 + daily_returns_series)) - 1
        packet['daily_perf']['benchmark_return'] = daily_returns_series[-1]
        packet['cumulative_perf']['benchmark_return'] = cumulative_return
        packet['cumulative_perf']['benchmark_annual_volatility'] = cumulative_annual_volatility


class AlphaBeta(object):
    """End of ArkQuant alpha and beta to the benchmark.
    """
    def __init__(self):
        self.return_series = None

    def start_of_simulation(self,
                            ledger,
                            benchmark_returns,
                            sessions):
        self.return_series = benchmark_returns[sessions]

    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions):
        risk = packet['cumulative_risk_metrics']
        benchmark_returns = self.return_series
        daily_value = ledger.portfolio.portfolio_daily_value
        daily_return_series = daily_value / daily_value.shift(1) - 1
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
    """A metric that reports an end of ArkQuant scalar or time series
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
                 _field_name=None,
                 risk_free=0.0,
                 required_return=0.0):

        self._function = function
        self.risk_free = risk_free
        self.required_return = required_return
        self._field_name = _field_name if _field_name else function.__name__
        self.return_series = None

    def start_of_simulation(self,
                            ledger,
                            benchmark_returns,
                            sessions):
        # 计算基准收益率
        self.return_series = benchmark_returns[sessions]

    def end_of_session(self,
                       packet,
                       ledger,
                       session_ix
                       ):
        # calculate daily return
        daily_value = ledger.portfolio.portfolio_daily_value
        daily_returns_series = daily_value / daily_value.shift(1) - 1
        return_series = self.return_series
        #
        daily_returns = daily_returns_series[daily_returns_series.index <= session_ix.strftime('%Y-%m-%d')]
        benchmark_returns = return_series[return_series.index <= session_ix.strftime('%Y-%m-%d')]
        res = self._function(
            daily_returns,
            benchmark_returns,
            self.risk_free,
            self.required_return
        )
        # print('ReturnsStatistic', self._field_name, res)
        if not np.isfinite(res):
            res = None
        packet['cumulative_risk_metrics'][self._field_name] = res
