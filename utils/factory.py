#
# Copyright 2016 Quantopian, Inc.
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


"""
Factory functions to prepare useful data.
"""
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

def create_trade(sid, price, amount, datetime, source_id="test_factory"):

    trade = Event()

    trade.source_id = source_id
    trade.type = DATASOURCE_TYPE.TRADE
    trade.sid = sid
    trade.dt = datetime
    trade.price = price
    trade.close_price = price
    trade.open_price = price
    trade.low = price * .95
    trade.high = price * 1.05
    trade.volume = amount

    return trade

class SpecificEquityTrades(object):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    count  : integer representing number of trades
    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """
    def __init__(self,
                 trading_calendar,
                 asset_finder,
                 sids,
                 start,
                 end,
                 delta,
                 count=500):

        self.trading_calendar = trading_calendar

        # Unpack config dictionary with default values.
        self.count = count
        self.start = start
        self.end = end
        self.delta = delta
        self.sids = sids
        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def __next__(self):
        return next(self.generator)

    def rewind(self):
        self.generator = self.create_fresh_generator()

    def update_source_id(self, gen):
        for event in gen:
            event.source_id = self.get_hash()
            yield event

    def create_fresh_generator(self):
        date_generator = date_gen(
            start=self.start,
            end=self.end,
            delta=self.delta,
            trading_calendar=self.trading_calendar,
        )
        return (
            create_trade(
                sid=sid,
                price=float(i % 10) + 1.0,
                amount=(i * 50) % 900 + 100,
                datetime=date,
            ) for (i, date), sid in itertools.product(
                enumerate(date_generator), self.sids
            )
        )

class SimulationParameters(object):
    def __init__(self,
                 start_session,
                 end_session,
                 trading_calendar,
                 capital_base=DEFAULT_CAPITAL_BASE,
                 emission_rate='daily',
                 data_frequency='daily',
                 arena='backtest'):

        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None, \
            "Must pass in trading calendar!"
        assert start_session <= end_session, \
            "Period start falls after period end."
        assert start_session <= trading_calendar.last_trading_session, \
            "Period start falls after the last known trading day."
        assert end_session >= trading_calendar.first_trading_session, \
            "Period end falls before the first known trading day."

        # chop off any minutes or hours on the given start and end dates,
        # as we only support session labels here (and we represent session
        # labels as midnight UTC).
        self._start_session = normalize_date(start_session)
        self._end_session = normalize_date(end_session)
        self._capital_base = capital_base

        self._emission_rate = emission_rate
        self._data_frequency = data_frequency

        # copied to algorithm's environment for runtime access
        self._arena = arena

        self._trading_calendar = trading_calendar

        if not trading_calendar.is_session(self._start_session):
            # if the start date is not a valid session in this calendar,
            # push it forward to the first valid session
            self._start_session = trading_calendar.minute_to_session_label(
                self._start_session
            )

        if not trading_calendar.is_session(self._end_session):
            # if the end date is not a valid session in this calendar,
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

    @property
    def emission_rate(self):
        return self._emission_rate

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
        self._arena = val

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
    @remember_last
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
                                 emission_rate='daily',
                                 trading_calendar=None):

    if not trading_calendar:
        trading_calendar = get_calendar("NYSE")

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
        emission_rate=emission_rate,
        trading_calendar=trading_calendar,
    )

    return sim_params


def get_next_trading_dt(current, interval, trading_calendar):
    next_dt = pd.Timestamp(current).tz_convert(trading_calendar.tz)

    while True:
        # Convert timestamp to naive before adding day, otherwise the when
        # stepping over EDT an hour is added.
        next_dt = pd.Timestamp(next_dt.replace(tzinfo=None))
        next_dt = next_dt + interval
        next_dt = pd.Timestamp(next_dt, tz=trading_calendar.tz)
        next_dt_utc = next_dt.tz_convert('UTC')
        if trading_calendar.is_open_on_minute(next_dt_utc):
            break
        next_dt = next_dt_utc.tz_convert(trading_calendar.tz)

    return next_dt_utc


def create_trade_history(sid, prices, amounts, interval, sim_params,
                         trading_calendar, source_id="test_factory"):
    trades = []
    current = sim_params.first_open

    oneday = timedelta(days=1)
    use_midnight = interval >= oneday
    for price, amount in zip(prices, amounts):
        if use_midnight:
            trade_dt = current.replace(hour=0, minute=0)
        else:
            trade_dt = current
        trade = create_trade(sid, price, amount, trade_dt, source_id)
        trades.append(trade)
        current = get_next_trading_dt(current, interval, trading_calendar)

    assert len(trades) == len(prices)
    return trades


def create_returns_from_range(sim_params):
    return pd.Series(index=sim_params.sessions,
                     data=np.random.rand(len(sim_params.sessions)))


def create_returns_from_list(returns, sim_params):
    return pd.Series(index=sim_params.sessions[:len(returns)],
                     data=returns)


def create_daily_trade_source(sids,
                              sim_params,
                              asset_finder,
                              trading_calendar):
    """
    creates trade_count trades for each sid in sids list.
    first trade will be on sim_params.start_session, and daily
    thereafter for each sid. Thus, two sids should result in two trades per
    day.
    """
    return create_trade_source(
        sids,
        timedelta(days=1),
        sim_params,
        asset_finder,
        trading_calendar=trading_calendar,
    )


def create_trade_source(sids,
                        trade_time_increment,
                        sim_params,
                        asset_finder,
                        trading_calendar):
    # If the sim_params define an end that is during market hours, that will be
    # used as the end of the data source
    if trading_calendar.is_open_on_minute(sim_params.end_session):
        end = sim_params.end_session
    # Otherwise, the last_close after the end_session is used as the end of the
    # data source
    else:
        end = sim_params.last_close

    args = tuple()
    kwargs = {
        'sids': sids,
        'start': sim_params.first_open,
        'end': end,
        'delta': trade_time_increment,
        'trading_calendar': trading_calendar,
        'asset_finder': asset_finder,
    }
    source = SpecificEquityTrades(*args, **kwargs)

    return source
