# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np, enum
from datetime import timedelta


DATASOURCE_TYPE = enum.Enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'TRANSACTION',
    'ORDER',
    'EMPTY',
    'DONE',
    'CUSTOM',
    'BENCHMARK',
    'COMMISSION',
    'CLOSE_POSITION'
)


class Event(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__.update(initial_values)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return hasattr(other, '__dict__') and self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)


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


def date_gen(start,
             end,
             trading_calendar,
             delta=timedelta(minutes=1),
             repeats=None):
    """
    Utility to generate a stream of dates.
    """
    daily_delta = not (delta.total_seconds()
                       % timedelta(days=1).total_seconds())
    cur = start
    if daily_delta:
        # if we are producing daily timestamps, we
        # use midnight
        cur = cur.replace(hour=0, minute=0, second=0,
                          microsecond=0)

    def advance_current(cur):
        """
        Advances the current dt skipping non market days and minutes.
        """
        cur = cur + delta

        currently_executing = \
            (daily_delta and (cur in trading_calendar.all_sessions)) or \
            (trading_calendar.is_open_on_minute(cur))

        if currently_executing:
            return cur
        else:
            if daily_delta:
                return trading_calendar.minute_to_session_label(cur)
            else:
                return trading_calendar.open_and_close_for_session(
                    trading_calendar.minute_to_session_label(cur)
                )[0]

    # yield count trade events, all on trading days, and
    # during trading hours.
    while cur < end:
        if repeats:
            for j in range(repeats):
                yield cur
        else:
            yield cur

        cur = advance_current(cur)


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
