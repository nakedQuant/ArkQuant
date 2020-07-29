#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import numpy as np ,pandas as pd,datetime
from toolz import partition_all
from functools import partial


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
            Start of period (inclusive) to produce metrics on
        end_session : pd.Timestamp
            End of period (inclusive) to produce metrics on
        algorithm_returns : pd.Series(pd.Timestamp -> float)
            Series of algorithm returns as of the end of each session
        benchmark_returns : pd.Series(pd.Timestamp -> float)
            Series of benchmark returns as of the end of each session
        algorithm_leverages : pd.Series(pd.Timestamp -> float)
            Series of algorithm leverages as of the end of each session

        Returns
        -------
        risk_metric : dict[str, any]
            Dict of metrics that with fields like:
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

        benchmark_period_returns = np.prod(benchmark_returns + 1) - 1
        algorithm_period_returns = np.prod(algorithm_returns + 1) - 1
        excess_period_returens = np.prod(excess_returns +1) - 1

        #组合胜率、超额胜率、
        absoulte_winrate = [algorithm_period_returns > 0].sum() / len(algorithm_period_returns)
        excess_winrate = (algorithm_period_returns > benchmark_period_returns).sum() / len(algorithm_period_returns)

        alpha, beta = ep.alpha_beta_aligned(
            algorithm_returns.values,
            benchmark_returns.values,
        )

        sharpe = ep.sharpe_ratio(algorithm_returns)

        # The consumer currently expects a 0.0 value for sharpe in period,
        # this differs from cumulative which was np.nan.
        # When factoring out the sharpe_ratio, the different return types
        # were collapsed into `np.nan`.
        # TODO: Either fix consumer to accept `np.nan` or make the
        # `sharpe_ratio` return type configurable.
        # In the meantime, convert nan values to 0.0
        if pd.isnull(sharpe):
            sharpe = 0.0

        sortino = ep.sortino_ratio(
            algorithm_returns.values,
            # 回撤
            _downside_risk=ep.downside_risk(algorithm_returns.values),
        )

        rval = {
            'algorithm_period_return': algorithm_period_returns,
            'benchmark_period_return': benchmark_period_returns,
            'excess_period_return': excess_period_returens,
            'absolute_winrate': absoulte_winrate,
            'excess_winrate': excess_winrate,
            'alpha': alpha,
            'beta': beta,
            'sharpe': sharpe,
            'sortino': sortino,
            'period_label': end_session.strftime("%Y-%m"),
            'trading_days': len(benchmark_returns),
            'algo_volatility': ep.annual_volatility(algorithm_returns),
            'benchmark_volatility': ep.annual_volatility(benchmark_returns),
            'max_drawdown': ep.max_drawdown(algorithm_returns.values),
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
                          # algorithm_leverages,
                          months_per):
        if months.size < months_per:
            return

        months_sequence = list(months)
        months_sequence.append(end_session)
        for period_timestamp in partition_all(months_per,months_sequence):
            try:
                start_time = period_timestamp[0]
                end_time = period_timestamp[-1]
            except:
                start_time = months_sequence[-2] + datetime.timedelta(days=1)
                end_time = period_timestamp[0]

            yield cls.risk_metric_period(
                start_session = start_time,
                end_session = end_time,
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
            closed = 'left'
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
                          sessions,
                          benchmark_source):
        packet.update(self.risk_report(
            algorithm_returns=ledger.daily_returns_series,
            benchmark_returns=benchmark_source.daily_returns(
                sessions[0],
                sessions[-1],
            ),
        ))