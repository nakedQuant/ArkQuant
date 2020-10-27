#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd
from datetime import datetime
from _calendar.trading_calendar import calendar
from trade import (
    DEFAULT_DELAY_BASE,
    DEFAULT_CAPITAL_BASE,
    DEFAULT_PER_CAPITAL_BASE
)


class SimulationParameters(object):
    """

    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    delay : int
        Transfer puts to calls (duals)
    """
    def __init__(self,
                 start_session,
                 end_session,
                 delay,
                 capital_base,
                 loan_base,
                 per_capital,
                 data_frequency,
                 benchmark):

        self._delay = delay
        self._loan = loan_base
        # per_capital used to calculate and split capital or position
        self.per_capital = per_capital
        self._capital_base = capital_base

        self._data_frequency = data_frequency
        self._benchmark = benchmark
        self._sessions = calendar.session_in_range(start_session, end_session)

    @property
    def capital_base(self):
        return self._capital_base

    @property
    def loan(self):
        return self._loan

    @property
    def data_frequency(self):
        return self._data_frequency

    @property
    def benchmark(self):
        return self._benchmark

    @property
    def delay(self):
        return self._delay

    @property
    def start_session(self):
        return pd.Timestamp(min(self._sessions))

    @property
    def end_session(self):
        return pd.Timestamp(max(self._sessions))

    @property
    # @remember_last #remember_last = weak_lru_cache(1)
    def sessions(self):
        return self._sessions

    def create_new(self, start_session, end_session, data_frequency=None):
        if data_frequency is None:
            data_frequency = self.data_frequency

        return SimulationParameters(
            start_session,
            end_session,
            capital_base=self.capital_base,
            data_frequency=data_frequency)

    def __repr__(self):
        return """
{class_name}(
    start_session={start_session},
    end_session={end_session},
    capital_base={capital_base},
    data_frequency={data_frequency}
)\
""".format(class_name=self.__class__.__name__,
           start_session=self.start_session,
           end_session=self.end_session,
           capital_base=self.capital_base,
           data_frequency=self.data_frequency)


def create_simulation_parameters(start=None,
                                 end=None,
                                 delay=DEFAULT_DELAY_BASE,
                                 per_capital=DEFAULT_PER_CAPITAL_BASE,
                                 capital_base=DEFAULT_CAPITAL_BASE,
                                 loan_base=0.0,
                                 data_frequency='daily',
                                 benchmark='000001'):

    if start is None:
        start = "{0}-01-01".format(2004)
    elif isinstance(start, str):
        start = start
    else:
        start = start.strftime('%Y-%m-%d')

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    elif isinstance(end, str):
        end = end
    else:
        end = end.strftime('%Y-%m-%d')

    sim_params = SimulationParameters(
        start_session=start,
        end_session=end,
        delay=delay,
        capital_base=capital_base,
        loan_base=loan_base,
        per_capital=per_capital,
        data_frequency=data_frequency,
        benchmark=benchmark
    )
    return sim_params


__all__ = ['create_simulation_parameters']
