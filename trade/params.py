#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd
from datetime import datetime
from _calendar.trading_calendar import calendar

# __func__ ---指向函数对象
DEFAULT_CAPITAL_BASE = 1e5


class SimulationParameters(object):
    def __init__(self,
                 start_session,
                 end_session,
                 capital_base=DEFAULT_CAPITAL_BASE,
                 data_frequency='daily'):

        self._capital_base = capital_base

        self._data_frequency = data_frequency

        self._sessions = calendar.session_in_range(start_session, end_session)

    @property
    def capital_base(self):
        return self._capital_base

    @property
    def data_frequency(self):
        return self._data_frequency

    @data_frequency.setter
    def data_frequency(self, val):
        self._data_frequency = val

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


def create_simulation_parameters(year=2004,
                                 start=None,
                                 end=None,
                                 capital_base=float("1.0e5"),
                                 data_frequency='daily'):

    if start is None:
        start = "{0}-01-01".format(year)
    elif isinstance(start, str):
        start = start
    else:
        start = start.strftime('%Y-%m-%d')

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    elif isinstance(start, str):
        end = start
    else:
        end = start.strftime('%Y-%m-%d')

    sim_params = SimulationParameters(
        start_session=start,
        end_session=end,
        capital_base=capital_base,
        data_frequency=data_frequency
    )
    return sim_params


__all__ = ['create_simulation_parameters']
