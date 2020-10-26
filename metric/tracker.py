#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import numpy as np, pandas as pd, datetime
from toolz import partition_all
from functools import partial
from metric.analyzers import (
                        sortino_ratio,
                        sharpe_ratio,
                        downside_risk,
                        annual_volatility,
                        max_drawdown,
                        )
from metric.exposure import alpha_beta_aligned


class MetricsTracker(object):
    """
    The algorithm's interface to the registered risk and performance
    metric.

    Parameters
    ----------
    sim_params : first_session, last_session, capital_base
        The params object
    metrics_sets : list[Metric]
        The metric to track.
    emission_rate : {'daily', 'minute'}
        How frequently should a performance packet be generated?
    """
    _hooks = (
        'start_of_simulation',
        'end_of_simulation',
        'start_of_session',
        'end_of_session',
    )

    def __init__(self,
                 ledger,
                 sim_params,
                 benchmark_rets,
                 metrics_sets,
                 emission_rate='daily'
                 ):
        self._ledger = ledger
        self._session = sim_params.session
        self._capital_base = sim_params.capital_base
        self._benchmark_returns = benchmark_rets
        self._emission_rate = emission_rate

        # bind all of the hooks from the passed metric objects.
        for hook in self._hooks:
            registered = []
            for metric in metrics_sets:
                try:
                    registered.append(getattr(metric, hook))
                except AttributeError:
                    pass

            def closing_over_loop_variables_is_hard(registered):
                def hook_implementation(*args, **kwargs):
                    for impl in registered:
                        impl(*args, **kwargs)
                return hook_implementation
            # 属性 --- 方法
            hook_implementation = closing_over_loop_variables_is_hard()
            hook_implementation.__name__ = hook
            # 属性 --- 方法
            setattr(self, hook, hook_implementation)

    def handle_start_of_simulation(self):
        self.start_of_simulation(
            self._ledger,
            self._benchmark_returns,
            self._sessions
        )

    def handle_market_open(self, session_label):
        """Handles the start of each session.

        Parameters
        ----------
        session_label : Timestamp
            The label of the session that is about to begin.
        """
        ledger = self._ledger
        # 账户初始化
        ledger.start_of_session(session_label)
        self.start_of_session(ledger)

    def handle_market_close(self, completed_session):
        """Handles the close of the given day.

        Parameters
        ----------
        completed_session : Timestamp
            The most recently completed nakedquant datetime.
        data_portal : DataPortal
            The current data portal.

        Returns
        -------
        A daily perf packet.
        """
        packet = {
            'period_start': self._sessions[0],
            'period_end': self._sessions[-1],
            'capital_base': self._capital_base,
            'emission_rate': self._emission_rate,
            'daily_perf': {},
            'cumulative_perf': {},
            'cumulative_risk_metrics': {},
        }
        ledger = self._ledger
        ledger.end_of_session()
        self.end_of_session(
            packet,
            ledger,
            completed_session,
        )
        return packet

    def handle_simulation_end(self):
        """
        When the nakedquant is complete, run the full period risk report
        and send it out on the results socket.
        """
        packet = {}
        self.end_of_simulation(
            packet,
            self._ledger,
            self._benchmark,
            self._sessions,
        )
        return packet


class _ClassicRiskMetrics(object):
    """
        Produces original risk packet.
    """
    @classmethod
    def risk_metric_period(cls,
                           start_session,
                           end_session,
                           algorithm_returns,
                           benchmark_returns):
        """
        Creates a dictionary representing the state of the risk report.

        Parameters
        ----------
        start_session : pd.Timestamp
            Start of period (inclusive) to produce metric on
        end_session : pd.Timestamp
            End of period (inclusive) to produce metric on
        algorithm_returns : pd.Series(pd.Timestamp -> float)
            Series of algorithm returns as of the end of each session
        benchmark_returns : pd.Series(pd.Timestamp -> float)
            Series of benchmark returns as of the end of each session
        # algorithm_leverages : pd.Series(pd.Timestamp -> float)
        #     Series of algorithm leverages as of the end of each session

        Returns
        -------
        risk_metric : dict[str, any]
            Dict of metric that with fields like:
                {
                    'algorithm_period_return': 0.0,
                    'benchmark_period_return': 0.0,
                    'treasury_period_return': 0,
                    'excess_return': 0.0,
                    'alpha': 0.0,
                    'beta': 0.0,
                    'sharpe': 0.0,
                    'sortino': 0.0,
                    'period_label': '1970-01',
                    'trading_days': 0,
                    'algo_volatility': 0.0,
                    'benchmark_volatility': 0.0,
                    'max_drawdown': 0.0,
                    'max_leverage': 0.0,
                }
        """
        algorithm_returns = algorithm_returns[
            (algorithm_returns.index >= start_session) &
            (algorithm_returns.index <= end_session)
        ]

        # Benchmark needs to be masked to the same dates as the algo returns
        benchmark_returns = benchmark_returns[
            (benchmark_returns.index >= start_session) &
            (benchmark_returns.index <= algorithm_returns.index[-1])
        ]

        excess_returns = algorithm_returns - benchmark_returns
        # 区间收益
        benchmark_period_returns = np.prod(benchmark_returns + 1) - 1
        algorithm_period_returns = np.prod(algorithm_returns + 1) - 1
        excess_period_returens = np.prod(excess_returns +1) - 1

        # 组合胜率、超额胜率
        absolute_winrate = [algorithm_period_returns > 0].sum() / len(algorithm_period_returns)
        excess_winrate = (algorithm_period_returns > benchmark_period_returns).sum() / len(algorithm_period_returns)

        alpha, beta = alpha_beta_aligned(
            algorithm_returns.values,
            benchmark_returns.values,
        )

        sharpe = sharpe_ratio(algorithm_returns)

        # The consumer currently expects a 0.0 value for sharpe in period,
        # this differs from cumulative which was np.nan.
        # When factoring out the sharpe_ratio, the different return types
        # were collapsed into `np.nan`.
        # TODO: Either fix consumer to accept `np.nan` or make the
        # `sharpe_ratio` return type configurable.
        # In the meantime, convert nan values to 0.0
        if pd.isnull(sharpe):
            sharpe = 0.0

        sortino = sortino_ratio(
            algorithm_returns.values,
            _downside_risk=downside_risk(algorithm_returns.values),
        )
        rval = {
            'algorithm_period_return': algorithm_period_returns,
            'benchmark_period_return': benchmark_period_returns,
            'excess_period_return': excess_period_returens,
            'absolute_winrate': absolute_winrate,
            'excess_winrate': excess_winrate,
            'alpha': alpha,
            'beta': beta,
            'sharpe': sharpe,
            'sortino': sortino,
            'period_label': end_session.strftime("%Y-%m"),
            'trading_days': len(benchmark_returns),
            'algo_volatility': annual_volatility(algorithm_returns),
            'benchmark_volatility': annual_volatility(benchmark_returns),
            'max_drawdown': max_drawdown(algorithm_returns.values),
        }
        # check if a field in rval is nan or inf, and replace it with None
        # except period_label which is always a str
        return {
            k: (
                None
                if k != 'period_label' and not np.isfinite(v) else
                v
            )
            for k, v in rval.items()
        }

    @classmethod
    def _periods_in_range(cls,
                          months,
                          end_session,
                          algorithm_returns,
                          benchmark_returns,
                          months_per):
        if months.size < months_per:
            return

        months_sequence = list(months)
        months_sequence.append(end_session)
        for period_timestamp in partition_all(months_per, months_sequence):
            try:
                start_time = period_timestamp[0]
                end_time = period_timestamp[-1]
            except KeyError:
                start_time = months_sequence[-2] + datetime.timedelta(days=1)
                end_time = period_timestamp[0]

            yield cls.risk_metric_period(
                start_session=start_time,
                end_session=end_time,
                algorithm_returns=algorithm_returns,
                benchmark_returns=benchmark_returns,
                # algorithm_leverages=algorithm_leverages,
            )

    @classmethod
    def risk_report(cls,
                    algorithm_returns,
                    benchmark_returns,
                    # algorithm_leverages
                    ):
        start_session = algorithm_returns.index[0]
        end_session = algorithm_returns.index[-1]
        months = pd.date_range(
            start=start_session,
            end=end_session,
            freq='M',
            tz='utc',
            closed='left'
        )

        periods_in_range = partial(
            cls._periods_in_range,
            months=months,
            end_session=end_session,
            algorithm_returns=algorithm_returns,
            benchmark_returns=benchmark_returns,
            # algorithm_leverages=algorithm_leverages,
        )

        return {
            'one_month': list(periods_in_range(months_per=1)),
            'three_month': list(periods_in_range(months_per=3)),
            'six_month': list(periods_in_range(months_per=6)),
            'twelve_month': list(periods_in_range(months_per=12)),
        }

    def end_of_simulation(self,
                          packet,
                          ledger,
                          benchmark_ret):
        packet.update(self.risk_report(
            algorithm_returns=ledger.daily_returns_series,
            benchmark_returns=benchmark_ret
        ))
        return packet
