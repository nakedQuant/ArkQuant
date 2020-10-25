# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, datetime, pandas as pd, pytz
from datetime import timedelta

MAX_MONTH_RANGE = 23
MAX_WEEK_RANGE = 5


def locate_pos(price, minutes, direction):
    try:
        loc = list(minutes.index[minutes == price])
        if not loc:
            # 当卖出价格大于bid价格才会成交，买入价格低于bid价格才会成交
            loc = list(minutes.index[minutes <= price]) if direction == '1' else \
                list(minutes.index[minutes >= price])
        return loc[0]
    except IndexError:
        print('price out of minutes')
        return None


def parse_date_str_series(format_str, tz, date_str_series):
    tz_str = str(tz)
    if tz_str == pytz.utc.zone:

        parsed = pd.to_datetime(
            date_str_series.values,
            format=format_str,
            utc=True,
            errors='coerce',
        )
    else:
        parsed = pd.to_datetime(
            date_str_series.values,
            format=format_str,
            errors='coerce',
        ).tz_localize(tz_str).tz_convert('UTC')
    return parsed


def naive_to_utc(ts):
    """
    Converts a UTC tz-naive timestamp to a tz-aware timestamp.
    """
    # Drop the nanoseconds field. warn=False suppresses the warning
    # that we are losing the nanoseconds; however, this is intended.
    return pd.Timestamp(ts.to_pydatetime(warn=False), tz='UTC')


def ensure_utc(time, tz='UTC'):
    """
    Normalize a time. If the time is tz-naive, assume it is UTC.
    """
    if not time.tzinfo:
        time = time.replace(tzinfo=pytz.timezone(tz))
    return time.replace(tzinfo=pytz.utc)


def normalize_date(frame):
    frame['year'] = frame['dates'] // 2048 + 2004
    frame['month'] = (frame['dates'] % 2048) // 100
    frame['day'] = (frame['dates'] % 2048) % 100
    frame['hour'] = frame['sub_dates'] // 60
    frame['minutes'] = frame['sub_dates'] % 60
    frame['ticker'] = frame.apply(lambda x: pd.Timestamp(
        datetime.datetime(int(x['year']), int(x['month']), int(x['day']),
                          int(x['hour']), int(x['minutes']))),
                            axis=1)
    # raw['timestamp'] = raw['ticker'].apply(lambda x: x.timestamp())
    # return frame.loc[:, ['ticker', 'open', 'high', 'low', 'close', 'amount', 'volume']]
    return frame


def _out_of_range_error(a, b=None, var='offset'):
    start = 0
    if b is None:
        end = a - 1
    else:
        start = a
        end = b - 1
    return ValueError(
        '{var} must be in between {start} and {end} inclusive'.format(
            var=var,
            start=start,
            end=end,
        )
    )


def _td_check(td):
    seconds = td.total_seconds()

    # 43200 seconds = 12 hours
    if 60 <= seconds <= 43200:
        return td
    else:
        raise ValueError('offset must be in between 1 minute and 12 hours, '
                         'inclusive.')


def _build_offset(offset, kwargs, default):
    """
    Builds the offset argument for event rules.
    """
    # Filter down to just kwargs that were actually passed.
    kwargs = {k: v for k, v in six.iteritems(kwargs) if v is not None}
    if offset is None:
        if not kwargs:
            return default  # use the default.
        else:
            return _td_check(datetime.timedelta(**kwargs))
    elif kwargs:
        raise ValueError('Cannot pass kwargs and an offset')
    elif isinstance(offset, datetime.timedelta):
        return _td_check(offset)
    else:
        raise TypeError("Must pass 'hours' and/or 'minutes' as keywords")


def _build_date(date, kwargs):
    """
    Builds the date argument for event rules.
    """
    if date is None:
        if not kwargs:
            raise ValueError('Must pass a date or kwargs')
        else:
            return datetime.date(**kwargs)

    elif kwargs:
        raise ValueError('Cannot pass kwargs and a date')
    else:
        return date


def _build_time(time, kwargs):
    """
    Builds the time argument for event rules.
    """
    tz = kwargs.pop('tz', 'UTC')
    if time:
        if kwargs:
            raise ValueError('Cannot pass kwargs and a time')
        else:
            return ensure_utc(time, tz)
    elif not kwargs:
        raise ValueError('Must pass a time or kwargs')
    else:
        return datetime.time(**kwargs)


def _time_to_micros(time):
    """Convert a time into microseconds since midnight.
    Parameters
    ----------
    time : datetime.time
        The time to convert.
    Returns
    -------
    us : int
        The number of microseconds since midnight.
    Notes
    -----
    This does not account for leap seconds or daylight savings.
    """
    seconds = time.hour * 60 * 60 + time.minute * 60 + time.second
    return 1000000 * seconds + time.microsecond


def timedelta_to_integral_seconds(delta):
    """
    Convert a pd.Timedelta to a number of seconds as an int.
    """
    return int(delta.total_seconds())


def timedelta_to_integral_minutes(delta):
    """
    Convert a pd.Timedelta to a number of minutes as an int.
    """
    return timedelta_to_integral_seconds(delta) // 60


def normalize_quarters(years, quarters):
    return years * 4 + quarters - 1


def split_normalized_quarters(normalized_quarters):
    years = normalized_quarters // 4
    quarters = normalized_quarters % 4
    return years, quarters + 1


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    aggregated_returns : pd.Series
    """

    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == 'weekly':
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == 'monthly':
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == 'quarterly':
        grouping = [lambda x: x.year, lambda x: int(np.math.ceil(x.month/3.))]
    elif convert_to == 'yearly':
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format('weekly', 'monthly', 'yearly')
        )

    return returns.groupby(grouping).apply(cumulate_returns)


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