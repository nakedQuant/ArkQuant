# -*- coding : utf-8 -*-
import operator as op,numpy as np
from toolz import groupby


class DailyFieldLedger(object):
    """基于字典 --- 无需返回值 ，局部变量里存在字典就OK了"""

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
                            ledger,
                            benchmark_source):
        self._start_of_simulation = self._get_ledger_field(ledger)

    def start_of_session(self, ledger):
        self._previous_day = self._get_ledger_field(ledger)

    def _end_of_period(self, sub_field, packet,ledger):
        packet_field = self._packet_field
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
                            ledger,
                            benchmark_source):
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
        packet['daily_perf']['returns'] = ledger.todays_returns
        packet['cumulative_perf']['returns'] = ledger.portfolio.returns


class CashFlow(object):
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """
    def start_of_simulation(self,
                            ledger,
                            benchmark_source):
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


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """
    def start_of_simulation(self,
                            ledger,
                            benchmark):
        # BenchmarkSource() --- benchmark_source
        self.benchmark_source = BenchmarkSource()
        self.benchmark = benchmark
        self._previous_cash_flow = 0.0

    def end_of_session(self,
                          packet,
                          ledger,
                          sessions,
                          session_ix):
        #计算基准收益率
        returns_series = self.benchmark_source.get_benchmark_returns(
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
        returns_series = BenchmarkSource().get_benchmark_returns(
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

