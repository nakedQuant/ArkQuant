# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from _calendar.trading_calendar import calendar
from trade import (
    SESSION_START,
    SESSION_END,
    BEFORE_TRADING_START
)


class MinuteSimulationClock(object):

    # before trading  , session start , session end 三个阶段
    def __init__(self, sim_params):

        self.sessions_nanos = sim_params.sessions
        self.trading_o_and_c = calendar.open_and_close_for_session(self.sessions_nanos)

    def __iter__(self):
        """
            If the clock property is not set, then create one based on frequency.
            session_minutes --- list , length --- 4
        """
        for session_label, session_minutes in zip(self.sessions_nanos, self.trading_o_and_c):
            yield session_label, BEFORE_TRADING_START
            for bts in session_minutes:
                if bts == pd.Timestamp(session_label) + pd.Timedelta(hours=9, minutes=30):
                    yield bts, SESSION_START
                else:
                    yield bts, SESSION_END
