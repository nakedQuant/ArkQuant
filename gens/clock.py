# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import pandas as pd

BAR = 0
SESSION_START = 1
SESSION_END = 2
MINUTE_END = 3
BEFORE_TRADING_START_BAR = 4

# before trading  , session start , session end 三个阶段


class MinuteSimulationClock(object):

    def __init__(self,
                 sessions,
                 trading_calendar):

        self.sessions_nanos = trading_calendar.session_in_range(*sessions, include=True)
        self.trading_o_and_c = trading_calendar.open_and_close_for_session(self.sim_params.sessions)
        self.minute_emission = 'minute'

    def __iter__(self):
        """
            If the clock property is not set, then create one based on frequency.
            session_minutes --- list , length --- 4
        """
        for session_label, session_minutes in zip(self.sessions_nanos, self.trading_o_and_c):
            yield session_label, BEFORE_TRADING_START_BAR
            bts_minute = pd.Timestamp(session_label) + 9 * 60 * 60 + 30 * 60
            if bts_minute == session_minutes[0]:
                yield bts_minute, SESSION_START
            bts_end = max(session_minutes[2:])
            yield bts_end, SESSION_END
