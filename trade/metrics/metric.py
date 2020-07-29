#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import operator as op,numpy as np
from toolz import groupby


# portfolio (portfolio_value , positions_exposure, positions_value , cash , gross_leverage , net_leverage), positions , daily_return

class DailyFieldLedger(object):

    def __init__(self,ledger_field,packet_field = None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.',1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix):
        field = self._packet_field
        packet['daily_perf'][field] = \
            self._get_ledger_field(ledger)


class StartOfPeriodLedgerField(object):
    """Keep track of the value of a ledger field at the start of the period.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger):
        self._start_of_simulation = self._get_ledger_field(ledger)

    def start_of_session(self, ledger):
        self._previous_day = self._get_ledger_field(ledger)

    def _end_of_period(self, sub_field, packet,ledger):
        packet_field = self._packet_field
        # start_of_simulation 不变的
        packet['cumulative_perf'][packet_field] = self._start_of_simulation
        packet[sub_field][packet_field] = self._previous_day

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       benchmark_sourc):
        self._end_of_period('daily_perf', packet,ledger)


class PNL(object):
    """Tracks daily and cumulative PNL.
    """
    def start_of_simulation(self,
                            ledger):
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
                       sessions,
                       session_ix,
                       benchmark_source):
        self._end_of_period('daily_perf', packet, ledger)


class Returns(object):
    """Tracks the daily and cumulative returns of the algorithm.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       benchmark_source):
        packet['daily_perf']['returns'] = ledger.daily_returns
        packet['cumulative_perf']['returns'] = ledger.portfolio.returns

def __init__(self, start_date=None, capital_base=0.0):
    self_ = MutableView(self)
    self_.cash_flow = 0.0
    self_.starting_cash = capital_base
    self_.portfolio_value = capital_base
    self_.pnl = 0.0
    self_.returns = 0.0
    self_.cash = capital_base
    self_.positions = Positions()
    self_.start_date = start_date
    self_.positions_value = 0.0
    self_.positions_exposure = 0.0



class CashFlow(object):
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """
    def start_of_simulation(self,
                            ledger):
        self._previous_cash_flow = 0.0

    def start_of_session(self,ledger):
        self._previous_cash_flow = ledger.portfolio.cash_flow

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix):
        cash_flow = ledger.portfolio.cash_flow
        packet['daily_perf']['capital_used'] = (
            cash_flow - self._previous_cash_flow
        )
        packet['cumulative_perf']['capital_used'] = cash_flow
        self._previous_cash_flow = cash_flow


class Transactions(object):
    """Tracks daily transactions.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix):
        packet['daily_perf']['transactions'] = ledger.get_transactions(session_ix)


class Positions(object):
    """Tracks daily positions.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix):
        packet['daily_perf']['positions'] = ledger.positions()


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """
    def start_of_simulation(self,
                            ledger,
                            benchmark):
        self.benchmark = benchmark
        self._previous_cash_flow = 0.0

    def end_of_session(self,
                          packet,
                          ledger,
                          sessions,
                          session_ix):
        #计算基准收益率
        returns_series = get_benchmark_returns(
            session_ix,
            self.benchmark
        )
        daily_returns_series = returns_series[sessions[0]:]
        #Series.expanding(self, min_periods=1, center=False, axis=0)
        cumulative_annual_volatility = (
            daily_returns_series.expanding(2).std(ddof=1) * np.sqrt(252)
        ).values[-1]

        cumulative_return = np.cumprod( 1+ daily_returns_series.values) -1

        packet['daily_perf']['benchmark_return'] = daily_returns_series[-1]
        packet['cumulative_perf']['benchmark_return'] = cumulative_return
        packet['cumulative_perf']['benchmark_annual_volatility'] = cumulative_annual_volatility


class AlphaBeta(object):
    """End of simulation alpha and beta to the benchmark.
    """
    def end_of_simulation(self,
                   packet,
                   ledger,
                   sessions,
                   benchmark):
        risk = packet['cumulative_risk_metrics']
        returns_series = get_benchmark_returns(
            sessions[-1],
            benchmark
        )
        benchmark_returns = returns_series[sessions[0]:]
        alpha, beta = ep.alpha_beta_aligned(
            ledger.daily_returns_array,
            benchmark_returns)

        if np.isnan(alpha):
            alpha = None
        if np.isnan(beta):
            beta = None

        risk['alpha'] = alpha
        risk['beta'] = beta


class ProbStatistics(object):
    """
        1、度量算法触发的概率（生成transaction)
        2、算法的胜率（产生正的收益概率）--- 当仓位完全退出时
    """
    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions,
                          benchmark):

        def calculate_rate(c_positions):
            dct = {}
            positive = [p.cost_basis > 0 for p in c_positions]
            negative = [p.cost_basis < 0 for p in c_positions]
            dct['win'] = len(positive)/ len(c_positions)
            dct['loss'] = len(negative)/ len(c_positions)
            return dct

        closed_positions = ledger.position_tracker.record_closed_position

        for origin, sliced_position in groupby(lambda x : x.asset_type.origin, closed_positions):
            packet['cumulative_risk_metrics']['hitrate']= calculate_rate(sliced_position)

        packet['cumulative_risk_metrics']['winrate'] = len([position.cost_basis > 0 for position in closed_positions])\
                                                                  /len(closed_positions)


class MaxLeverage(object):
    """Tracks the maximum account leverage.
    """
    def start_of_simulation(self, *args):
        self._max_leverage = 0.0

    def end_of_session(self,
                   packet,
                   ledger,
                   sessions,
                   session_ix):
        self._max_leverage = max(self._max_leverage, ledger.account.leverage)
        packet['cumulative_risk_metrics']['max_leverage'] = self._max_leverage


class NumTradingDays(object):
    """Report the number of trading days.
    """
    def start_of_simulation(self, *args):
        self._num_trading_days = 0

    def start_of_session(self,*args):
        self._num_trading_days += 1

    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions,
                          benchmark):
        packet['cumulative_risk_metrics']['trading_days'] = \
            self._num_trading_days

    # def end_of_bar(self,
    #                packet,
    #                ledger,
    #                dt,
    #                session_ix,
    #                data_portal):
    #     packet['cumulative_risk_metrics']['trading_days'] = (
    #         self._num_trading_days
    #     )
    #
    # end_of_session = end_of_bar


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

    def end_of_bar(self, packet, *args):
        packet['cumulative_risk_metrics'][self._field] = self._value

    def end_of_session(self, packet, *args):
        packet['cumulative_risk_metrics'][self._field] = self._value


class PeriodLabel(object):
    """Backwards compat, please kill me.
    """
    def start_of_session(self, ledger, session, data_portal):
        self._label = session.strftime('%Y-%m')

    def end_of_bar(self, packet, *args):
        packet['cumulative_risk_metrics']['period_label'] = self._label

    end_of_session = end_of_bar


class SQN(object):
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
    def __init__(self, function, field_name=None):
        if field_name is None:
            field_name = function.__name__

        self._function = function
        self._field_name = field_name

    def end_of_session(self,
                   packet,
                   ledger,
                   sessions,
                   session_ix):
        # res = self._function(ledger.daily_returns_array[:session_ix + 1])
        res = self._function(ledger.daily_returns_array)
        if not np.isfinite(res):
            res = None
        packet['cumulative_risk_metrics'][self._field_name] = res



class MetricsTracker(object):
    """
    The algorithm's interface to the registered risk and performance
    metrics.

    Parameters
    ----------
    trading_calendar : TrandingCalendar
        The trading calendar used in the simulation.
    first_session : pd.Timestamp
        The label of the first trading session in the simulation.
    last_session : pd.Timestamp
        The label of the last trading session in the simulation.
    capital_base : float
        The starting capital for the simulation.
    metrics : list[Metric]
        The metrics to track.
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
                 first_session,
                 last_session,
                 trading_calendar,
                 capital_base,
                 metrics,
                 ):
        self._sessions  = trading_calendar.sessions_in_range(
            first_session,
            last_session,
        )
        self._ledger = ledger

        self._capital_base = capital_base

        self._first_session = first_session
        self._last_session = last_session

        # bind all of the hooks from the passed metrics objects.
        for hook in self._hooks:
            registered = []
            for metric in metrics:
                try:
                    registered.append(getattr(metric, hook))
                except AttributeError:
                    pass

            def closing_over_loop_variables_is_hard(registered):
                def hook_implementation(*args, **kwargs):
                    for impl in registered:
                        impl(*args, **kwargs)

                return hook_implementation
            #属性 --- 方法
            hook_implementation = closing_over_loop_variables_is_hard()
            hook_implementation.__name__ = hook
            # 属性 --- 方法
            setattr(self, hook, hook_implementation)

    def handle_start_of_simulation(self, benchmark):
        self._benchmark = benchmark

        self.start_of_simulation(
            self._ledger,
            benchmark,
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
        #执行metrics --- start_of_session
        self.start_of_session(ledger, session_label)
        self._current_session = session_label

    def handle_market_close(self):
        """Handles the close of the given day.

        Parameters
        ----------
        dt : Timestamp
            The most recently completed simulation datetime.
        data_portal : DataPortal
            The current data portal.

        Returns
        -------
        A daily perf packet.
        """
        completed_session = self._current_session

        packet = {
            'period_start': self._first_session,
            'period_end': self._last_session,
            'capital_base': self._capital_base,
            'daily_perf': {},
            'cumulative_perf': {},
            'cumulative_risk_metrics': {},
        }
        ledger = self._ledger
        ledger.end_of_session(completed_session)
        self.end_of_session(
            packet,
            ledger,
            self._sessions,
            completed_session,
            self._benchmark_source
        )
        return packet

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """
        packet = {}
        self.end_of_simulation(
            packet,
            self._ledger,
            self._sessions,
            self._benchmark,
        )
        return packet
