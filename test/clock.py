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

import numpy as np ,pandas as pd

_nanos_in_minute = 60000000000
NANOS_IN_MINUTE = _nanos_in_minute

BAR = 0
SESSION_START = 1
SESSION_END = 2
MINUTE_END = 3
BEFORE_TRADING_START_BAR = 4


# before trading  , session start , session end 三个阶段

# for dt, action in self.clock:
#     if action == BEFORE_TRADING_START_BAR:
#         # algo.before_trading_start(self.current_data)
#         metrics_tracker.handle_market_open(dt)
#     elif action == SESSION_START:
#         once_a_day(dt)
#     elif action == SESSION_END:
#         # End of the session.
#         yield daily_perf_metrics

def _create_clock(self):
    """
    If the clock property is not set, then create one based on frequency.
    """
    trading_o_and_c = self.trading_calendar.schedule.ix[
        self.sim_params.sessions]
    market_closes = trading_o_and_c['market_close']
    minutely_emission = False

    if self.sim_params.data_frequency == 'minute':
        market_opens = trading_o_and_c['market_open']
        minutely_emission = self.sim_params.emission_rate == "minute"

        # The calendar's execution times are the minutes over which we
        # actually want to run the clock. Typically the execution times
        # simply adhere to the market open and close times. In the case of
        # the futures calendar, for example, we only want to simulate over
        # a subset of the full 24 hour calendar, so the execution times
        # dictate a market open time of 6:31am US/Eastern and a close of
        # 5:00pm US/Eastern.
        execution_opens = \
            self.trading_calendar.execution_time_from_open(market_opens)
        execution_closes = \
            self.trading_calendar.execution_time_from_close(market_closes)
    else:
        # in daily mode, we want to have one bar per session, timestamped
        # as the last minute of the session.
        execution_closes = \
            self.trading_calendar.execution_time_from_close(market_closes)
        execution_opens = execution_closes

    # FIXME generalize these values
    before_trading_start_minutes = days_at_time(
        self.sim_params.sessions,
        time(8, 45),
        "US/Eastern"
    )

    return MinuteSimulationClock(
        self.sim_params.sessions,
        execution_opens,
        execution_closes,
        before_trading_start_minutes,
        minute_emission=minutely_emission,
    )


class MinuteSimulationClock:

    def __init__(self,
                 sessions,
                 market_opens,
                 market_closes,
                 before_trading_start_minutes,
                 minute_emission=False):
        self.minute_emission = minute_emission

        self.market_opens_nanos = market_opens.values.astype(np.int64)
        self.market_closes_nanos = market_closes.values.astype(np.int64)
        self.sessions_nanos = sessions.values.astype(np.int64)
        self.bts_nanos = before_trading_start_minutes.values.astype(np.int64)

        self.minutes_by_session = self.calc_minutes_by_session()

    def calc_minutes_by_session(self):
        minutes_by_session = {}
        for session_idx, session_nano in enumerate(self.sessions_nanos):
            minutes_nanos = np.arange(
                self.market_opens_nanos[session_idx],
                self.market_closes_nanos[session_idx] + _nanos_in_minute,
                _nanos_in_minute
            )
            minutes_by_session[session_nano] = pd.to_datetime(
                minutes_nanos, utc=True, box=True
            )
        return minutes_by_session

    def __iter__(self):
        minute_emission = self.minute_emission
        # yield 中断开始 --- session start , session minutes  , session end

        for idx, session_nano in enumerate(self.sessions_nanos):
            yield pd.Timestamp(session_nano, tz='UTC'), SESSION_START

            bts_minute = pd.Timestamp(self.bts_nanos[idx], tz='UTC')
            regular_minutes = self.minutes_by_session[session_nano]

            if bts_minute > regular_minutes[-1]:
                # before_trading_start is after the last close,
                # so don't emit it
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes,
                    minute_emission
                ):
                    yield minute, evt
            else:
                # we have to search anew every session, because there is no
                # guarantee that any two session start on the same minute
                bts_idx = regular_minutes.searchsorted(bts_minute)

                # emit all the minutes before bts_minute
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes[0:bts_idx],
                    minute_emission
                ):
                    yield minute, evt

                yield bts_minute, BEFORE_TRADING_START_BAR

                # emit all the minutes after bts_minute
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes[bts_idx:],
                    minute_emission
                ):
                    yield minute, evt

            yield regular_minutes[-1], SESSION_END

    def _get_minutes_for_list(self, minutes, minute_emission):
        for minute in minutes:
            yield minute, BAR
            if minute_emission:
                yield minute, MINUTE_END
