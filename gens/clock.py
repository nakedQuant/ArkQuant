#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        self.sessions_nanos = trading_calendar.session_in_range(sessions)
        self.trading_o_and_c = trading_calendar.open_and_close_for_session(self.sim_params.sessions)
        self.minute_emission = 'minute'

    def __iter__(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        for session_label , session_minutes in zip(self.trading_days,self.trading_o_and_c):
            yield session_label, BEFORE_TRADING_START_BAR
            bts_minute = pd.Timestamp(session_label) + 9 * 60 * 60 + 30 * 60
            if bts_minute == session_minutes[0]:
                yield bts_minute , SESSION_START
            bts_end = max(session_minutes[2:])
            yield bts_end , SESSION_END
