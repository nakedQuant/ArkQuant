#
# Copyright 2013 Quantopian, Inc.
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
import pandas as pd , pytz , numpy as np , sqlalchemy as sa

from datetime import datetime
from dateutil import rrule
from functools import partial

# from trading_calendars import (
#     clear_calendars,
#     deregister_calendar,
#     get_calendar,
#     register_calendar,
#     register_calendar_alias,
#     register_calendar_type,
#     TradingCalendar,
# )

# 元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日
# 春节
spring = ['19900127','19910215','19920204','19930123','19940210',
            '19950131','19960219','19970207','19980128','19990216',
            '20000205','20010124','20020212','20030201','20040122',
            '20050209','20060129','20070218','20080207','20090126',
            '20100214','20110203','20120123','20130210','20140131',
            '20150219','20160208','20170128','20180216','20190205',
            '20200125','20210212','20220201','20230122','20240210',
            '20250129','20260217','20270206','20280126','20290213',
            '20300203','20310123','20320211','20330131','20340219',
            '20350208','20360128','20370215','20380204','20390124',
            '20400212','20410201','20420122','20430210','20440130',
            '20450217','20460206','20470126','20480214','20490202',
            '20500123','20510211','20520201','20530219','20540208',
            '20550128','20560215','20570204','20580124','20590212',
            '20600202','20610121','20620209','20630129','20640217',
            '20650205','20660126','20670214','20680203','20690123']
# 中秋；
autumn = ['19901003','19910922','19920911','19930930','19940920',
          '19950909','19960927','19970916','19981005','19990924',
          '20000912','20011001', '20020921','20030911','20040928',
          '20050918','20061006','20070925','20080914','20091003',
          '20100922','20110912','20120930','20130919', '20140908',
          '20150927','20160915','20171004','20180924','20190913',
          '20201001','20210921','20220910', '20230929','20240917',
          '20251006','20260925', '20270915','20281003','20290922',
          '20300912','20311001','20320919', '20330908','20340927',
          '20350916','20361004','20370924','20380913','20391002',
          '20400920','20410910','20420928','20430917','20441005',
          '20450925','20460915''20471004','20480922','20490911','20500930',]

fixed_holiday_names = frozenset(('new_year','spring','tomb','labour','autumn','national'))

class Calendar(object):

    def __init__(self,conn):
        self.conn = conn
        self._init_calendar()
        self._fixed_holiday()

    def _init_calendar(self):
        """获取交易日"""
        table = self.tables['trading_calendar']
        sql = sa.select([table.c.trade_dt])
        rp = self.conn.execute(sql)
        self.trading_days = [r.trade_dt for r in rp]

    def shift_calendar(self,dt,window):
        index = np.searchsorted(self.trading_days,dt)
        if index + window < 0:
            raise ValueError('out of trading_calendar range')
        return self.trading_days[index + window]

    def calculate_window_size(self, start_dt, end_dt):
        idx_s = np.searchsorted(self.trading_days, start_dt)
        idx_e = np.searchsorted(self.trading_days,end_dt)
        return idx_e - idx_s

    @property
    def _fixed_holiday(self,):
        non_trading_rules = dict()
        non_trading_rules['spring'] = spring
        non_trading_rules['autumn'] = autumn
        tz = pytz.timezone('Asia/Shanghai')
        start = pd.Timestamp(min(self.trading_days), tz=tz)
        end = pd.Timestamp(max(self.trading_days), tz=tz)

        new_year = rrule.rrule(
            rrule.YEARLY,
            byyearday= 1,
            cache = True,
            dstart = start,
            until = end
        )
        non_trading_rules.update({'new_year':new_year})

        april_4 = rrule.rrule(
            rrule.YEARLY,
            bymonth= 4,
            bymonthday= 4,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'tomb':april_4})

        may_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=5,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'labour':may_day})

        national_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=10,
            bymonthday= 1,
            cache=True,
            dtstart=start,
            until=end
        )

        non_trading_rules.update({'national':national_day})
        return non_trading_rules

    def get_last_trading_day_before_holiday(self,holiday_name):
        if holiday_name not in fixed_holiday_names:
            raise ValueError('unidentified holiday name')
        holiday_days = self._fixed_holiday[holiday_name]
        linear_loc = np.searchsorted(self.trading_days,holiday_days)
        return self.trading_days[linear_loc]

    def get_open_and_close(day, early_closes):

        market_open = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=31),
            tz='US/Eastern').tz_convert('UTC')
        # 1 PM if early close, 4 PM otherwise
        close_hour = 13 if day in early_closes else 16
        market_close = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=close_hour),
            tz='US/Eastern').tz_convert('UTC')

        return market_open, market_close

    def get_open_and_closes(trading_days, early_closes, get_open_and_close):
        open_and_closes = pd.DataFrame(index=trading_days,
                                       columns=('market_open', 'market_close'))

        get_o_and_c = partial(get_open_and_close, early_closes=early_closes)

        open_and_closes['market_open'], open_and_closes['market_close'] = \
            zip(*open_and_closes.index.map(get_o_and_c))

        return open_and_closes

    def get_early_closes(start, end):
        """
            熔断机制
        """
        start = max(start, datetime(1993, 1, 1, tzinfo=pytz.utc))
        end = max(end, datetime(1993, 1, 1, tzinfo=pytz.utc))

        early_close_rules = []

        day_after_thanksgiving = rrule.rrule(
            rrule.MONTHLY,
            bymonth=11,
            # 4th Friday isn't correct if month starts on Friday, so restrict to
            # day range:
            byweekday=(rrule.FR),
            bymonthday=range(23, 30),
            cache=True,
            dtstart=start,
            until=end
        )

from toolz import partition_all


def compute_date_range_chunks(sessions, start_date, end_date, chunksize):
    """Compute the start and end dates to run a pipeline for.

    Parameters
    ----------
    sessions : DatetimeIndex
        The available dates.
    start_date : pd.Timestamp
        The first date in the pipeline.
    end_date : pd.Timestamp
        The last date in the pipeline.
    chunksize : int or None
        The size of the chunks to run. Setting this to None returns one chunk.

    Returns
    -------
    ranges : iterable[(np.datetime64, np.datetime64)]
        A sequence of start and end dates to run the pipeline for.
    """
    if start_date not in sessions:
        raise KeyError("Start date %s is not found in calendar." %
                       (start_date.strftime("%Y-%m-%d"),))
    if end_date not in sessions:
        raise KeyError("End date %s is not found in calendar." %
                       (end_date.strftime("%Y-%m-%d"),))
    if end_date < start_date:
        raise ValueError("End date %s cannot precede start date %s." %
                         (end_date.strftime("%Y-%m-%d"),
                          start_date.strftime("%Y-%m-%d")))

    if chunksize is None:
        return [(start_date, end_date)]

    start_ix, end_ix = sessions.slice_locs(start_date, end_date)
    return (
        (r[0], r[-1]) for r in partition_all(
            chunksize, sessions[start_ix:end_ix]
        )
    )
