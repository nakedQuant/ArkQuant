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

start = pd.Timestamp('1990-01-01', tz='UTC')
end_base = pd.Timestamp('today', tz='UTC')
end = end_base + pd.Timedelta(days=365)

# 元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日
# 春节
spring = ['1990-1-27','1991-2-15','1992-2-4','1993-1-23','1994-2-10',
            '1995-1-31','1996-2-19','1997-2-7','1998-1-28','1999-2-16',
            '2000-2-5','2001-1-24','2002-2-12','2003-2-1','2004-1-22',
            '2005-2-9','2006-1-29','2007-2-18','2008-2-7','2009-1-26',
            '2010-2-14','2011-2-3','2012-1-23','2013-2-10','2014-1-31',
            '2015-2-19','2016-2-8','2017-1-28','2018-2-16','2019-2-5',
            '2020-1-25','2021-2-12','2022-2-1','2023-1-22','2024-2-10',
            '2025-1-29','2026-2-17','2027-2-6','2028-1-26','2029-2-13',
            '2030-2-3','2031-1-23','2032-2-11','2033-1-31','2034-2-19',
            '2035-2-8','2036-1-28','2037-2-15','2038-2-4','2039-1-24',
            '2040-2-12','2041-2-1','2042-1-22','2043-2-10','2044-1-30',
            '2045-2-17','2046-2-6','2047-1-26','2048-2-14','2049-2-2',
            '2050-1-23','2051-2-11','2052-2-1','2053-2-19','2054-2-8',
            '2055-1-28','2056-2-15','2057-2-4','2058-1-24','2059-2-12',
            '2060-2-2','2061-1-21','2062-2-9','2063-1-29','2064-2-17',
            '2065-2-5','2066-1-26','2067-2-14','2068-2-3','2069-1-23']
# 中秋；
autumn = ['1990-10-3','1991-9-22','1992-9-11','1993-9-30','1994-9-20',
          '1995-9-9','1996-9-27','1997-9-16','1998-10-5','1999-9-24',
          '2000-9-12','2001-10-1', '2002-9-21','2003-9-11','2004-9-28',
          '2005-9-18','2006-10-6','2007-9-25','2008-9-14','2009-10-3',
          '2010-9-22','2011-9-12','2012-9-30','2013-9-19', '2014-9-8',
          '2015-9-27','2016-9-15','2017-10-4','2018-9-24','2019-9-13',
          '2020-10-1','2021-9-21','2022-9-10', '2023-9-29','2024-9-17',
          '2025-10-6','2026-9-25', '2027-9-15','2028-10-3','2029-9-22',
          '2030-9-12','2031-10-1','2032-9-19', '2033-9-8','2034-9-27',
          '2035-9-16','2036-10-4','2037-9-24','2038-9-13','2039-10-2',
          '2040-9-20', '2041-9-10','2042-9-28','2043-9-17','2044-10-5',
          '2045-9-25', '2046-9-15''2047-10-4','2048-9-22','2049-9-11','2050-9-30',]


class Calendar(object):

    def __init__(self,conn):
        self.conn = conn
        self._init_calendar()

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

    def get_non_trading_days(start, end):
        """
            元旦 1-1
            春节
            清明节 4-4
            劳动节 5-1
            中秋节
            国庆节 10-1
        """
        non_trading_rules = []

        weekends = rrule.rrule(
            rrule.YEARLY,
            byweekday=(rrule.SA, rrule.SU),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(weekends)

        new_years = rrule.rrule(
            rrule.MONTHLY,
            byyearday=1,
            cache=True,
            dtstart=start,
            until=end
        )

        good_friday = rrule.rrule(
            rrule.DAILY,
            byeaster=-2,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(good_friday)

        memorial_day = rrule.rrule(
            rrule.MONTHLY,
            bymonth=5,
            byweekday=(rrule.MO(-1)),
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.append(memorial_day)


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

        # Not included here are early closes prior to 1993
        # or unplanned early closes

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
        early_close_rules.append(day_after_thanksgiving)
        # return pd.DatetimeIndex(early_closes)