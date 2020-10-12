#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import pandas as pd
from datetime import datetime
from _calendar.trading_calendar import calendar

__all__ = ['create_simulation_parameters']

# __func__ ---指向函数对象
DEFAULT_CAPITAL_BASE = 1e5


class SimulationParameters(object):
    def __init__(self,
                 start_session,
                 end_session,
                 trading_calendar,
                 benchmark,
                 capital_base=DEFAULT_CAPITAL_BASE,
                 # emission_rate='daily',
                 # arena='backtest',
                 data_frequency='daily'):

        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None, \
            "Must pass in trading _calendar!"
        assert start_session <= end_session, \
            "Period start falls after period end."
        assert start_session <= trading_calendar.last_trading_session, \
            "Period start falls after the last known trading day."
        assert end_session >= trading_calendar.first_trading_session, \
            "Period end falls before the first known trading day."

        # chop off any minutes or hours on the given start and end dates,
        # as we only support session labels here (and we represent session
        # labels as midnight UTC).
        # self._start_session = normalize_date(start_session)
        # self._end_session = normalize_date(end_session)

        self._capital_base = capital_base

        # self._emission_rate = emission_rate
        self._data_frequency = data_frequency

        # copied to algorithm's environment for runtime access
        # self._arena = arena

        self._trading_calendar = trading_calendar

        if not trading_calendar.is_session(self._start_session):
            # if the start date is not a valid session in this _calendar,
            # push it forward to the first valid session
            self._start_session = trading_calendar.minute_to_session_label(
                self._start_session
            )

        if not trading_calendar.is_session(self._end_session):
            # if the end date is not a valid session in this _calendar,
            # pull it backward to the last valid session before the given
            # end date.
            self._end_session = trading_calendar.minute_to_session_label(
                self._end_session, direction="previous"
            )

        self._first_open = trading_calendar.open_and_close_for_session(
            self._start_session
        )[0]
        self._last_close = trading_calendar.open_and_close_for_session(
            self._end_session
        )[1]

    @property
    def capital_base(self):
        return self._capital_base

    # @property
    # def emission_rate(self):
    #     return self._emission_rate

    @property
    def data_frequency(self):
        return self._data_frequency

    @data_frequency.setter
    def data_frequency(self, val):
        self._data_frequency = val

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, val):
        self.arena = val

    @property
    def start_session(self):
        return self._start_session

    @property
    def end_session(self):
        return self._end_session

    @property
    def first_open(self):
        return self._first_open

    @property
    def last_close(self):
        return self._last_close

    @property
    def trading_calendar(self):
        return self._trading_calendar

    @property
    # @remember_last #remember_last = weak_lru_cache(1)
    def sessions(self):
        return self._trading_calendar.sessions_in_range(
            self.start_session,
            self.end_session
        )

    def create_new(self, start_session, end_session, data_frequency=None):
        if data_frequency is None:
            data_frequency = self.data_frequency

        return SimulationParameters(
            start_session,
            end_session,
            self._trading_calendar,
            capital_base=self.capital_base,
            emission_rate=self.emission_rate,
            data_frequency=data_frequency,
            arena=self.arena
        )

    def __repr__(self):
        return """
{class_name}(
    start_session={start_session},
    end_session={end_session},
    capital_base={capital_base},
    data_frequency={data_frequency},
    emission_rate={emission_rate},
    first_open={first_open},
    last_close={last_close},
    trading_calendar={trading_calendar}
)\
""".format(class_name=self.__class__.__name__,
           start_session=self.start_session,
           end_session=self.end_session,
           capital_base=self.capital_base,
           data_frequency=self.data_frequency,
           emission_rate=self.emission_rate,
           first_open=self.first_open,
           last_close=self.last_close,
           trading_calendar=self._trading_calendar)


def create_simulation_parameters(year=2006,
                                 start=None,
                                 end=None,
                                 capital_base=float("1.0e5"),
                                 num_days=None,
                                 data_frequency='daily',
                                 # emission_rate='daily',
                                 trading_calendar=None):

    if not trading_calendar:
        trading_calendar = calendar

    if start is None:
        start = pd.Timestamp("{0}-01-01".format(year), tz='UTC')
    elif type(start) == datetime:
        start = pd.Timestamp(start)

    if end is None:
        if num_days:
            start_index = trading_calendar.all_sessions.searchsorted(start)
            end = trading_calendar.all_sessions[start_index + num_days - 1]
        else:
            end = pd.Timestamp("{0}-12-31".format(year), tz='UTC')
    elif type(end) == datetime:
        end = pd.Timestamp(end)

    sim_params = SimulationParameters(
        start_session=start,
        end_session=end,
        capital_base=capital_base,
        data_frequency=data_frequency,
        # emission_rate=emission_rate,
        trading_calendar=trading_calendar,
    )

    return sim_params

