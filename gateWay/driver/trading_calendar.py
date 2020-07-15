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
import pandas as pd, pytz, numpy as np
from datetime import datetime
from dateutil import rrule
from toolz import partition_all

__all__ = ['TradingCalendar']

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
          '20450925','20460915''20471004','20480922','20490911','20500930']

fixed_holiday_names = frozenset(('new_year','spring','tomb','labour','autumn','national'))


class TradingCalendar (object):
    """
    元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日 春节 中秋
    """

    def __init__(self):
        self._fixed_holiday()

    def _init_calendar_cache(self):
        """获取交易日"""

    @property
    def calendar(self):
        pass

    def _roll_forward(self, dt,window):
        """
        Given a date, align it to the calendar of the pipeline's domain.
        dt = pd.Timestamp(dt, tz='UTC')

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
        """
        pos = self.all_sessions.searchsorted(dt)
        try:
            loc = pos if self.all_sessions[pos] == dt else pos -1
            forward_dt = self.all_sessions[loc - window]
            return self.all_sessions[forward_dt]
        except IndexError:
            raise ValueError(
                "Date {} was past the last session for domain {}. "
                "The last session for this domain is {}.".format(
                    dt.date(),
                    self,
                    self.all_sessions[-1].date()
                )
            )

    def _compute_date_range_slice(self, start_date, end_date):
        # Get the index of the start of dates for ``start_date``.
        start_ix = self.dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = self.dates.searchsorted(end_date, side='right')

        return slice(start_ix, end_ix)

    def session_in_range(self,start_date,end_date):
        if end_date < start_date:
            raise ValueError("End date %s cannot precede start date %s." %
                             (end_date.strftime("%Y-%m-%d"),
                              start_date.strftime("%Y-%m-%d")))
        idx_s = np.searchsorted(self.all_sessions, start_date)
        idx_e = np.searchsorted(self.all_sessions,end_date)
        end = idx_e -1 if self.all_sessions[idx_e] > end_date else idx_e
        return self.all_sessions[idx_s:end]

    def session_in_window(self,end_date,window):
        start_date = self._roll_forward(end_date,window)
        if end_date < start_date:
            raise ValueError("End date %s cannot precede start date %s." %
                             (end_date.strftime("%Y-%m-%d"),
                              start_date.strftime("%Y-%m-%d")))
        idx_s = np.searchsorted(self.all_sessions, start_date)
        idx_e = np.searchsorted(self.all_sessions,end_date)
        end = idx_e -1 if self.all_sessions[idx_e] > end_date else idx_e
        return self.all_sessions[idx_s:end]

    def minutes_in_sessions(self,dts):
        minutes_session = map(lambda x : list(range(pd.Timestamp(x) + 9*60*60 + 30*60,
                                                    pd.Timestamp(x) + 15*60*60 + 1)),
                                dts)
        return minutes_session

    def open_and_close_for_session(self,dts):
        opens = [ pd.Timestamp(dt) + 9 * 60 * 60 + 30 * 60 for dt in dts]
        closes = [ pd.Timestamp(dt) + 15 * 60 * 60 for dt in dts]
        o_c = zip(opens,closes)
        return list(o_c)

    def compute_range_chunks(self,start_date, end_date, chunksize):
        """Compute the start and end dates to run a pipeline for.

        Parameters
        ----------
        start_date : pd.Timestamp
            The first date in the pipeline.
        end_date : pd.Timestamp
            The last date in the pipeline.
        chunksize : int or None
            The size of the chunks to run. Setting this to None returns one chunk.
        """
        sessions = self.session_in_range(start_date,end_date)
        return (
            (r[0], r[-1]) for r in partition_all(
            chunksize, sessions
        )
        )

    @property
    def _fixed_holiday(self,):
        non_trading_rules = dict()
        non_trading_rules['spring'] = spring
        non_trading_rules['autumn'] = autumn
        tz = pytz.timezone('Asia/Shanghai')
        start = pd.Timestamp(min(self.all_sessions), tz=tz)
        end = pd.Timestamp(max(self.all_sessions), tz=tz)

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

    def get_trading_day_near_holiday(self,holiday_name,forward = True):
        if holiday_name not in fixed_holiday_names:
            raise ValueError('unidentified holiday name')
        holiday_days = self._fixed_holiday[holiday_name]
        idx_list = [ np.searchsorted(self.all_sessions, t)  for t in holiday_days]
        if forward:
            trading_list = self.all_sessions[list(map(lambda x : x -1,idx_list))]
        else:
            trading_list = self.all_sessions[idx_list]
        return trading_list

    def get_open_and_close(day):

        market_open = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=30),
            tz='Asia/Shanghai')
        market_close = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=15,
                minute=0),
                tz='Asia/Shanghai')
        return market_open, market_close

    def get_early_close_days(self):
        """
            circuitBreaker --- 熔断机制 2016-01-01 2016-01-07
        """
        early_close_days = self.session_in_range('2016-01-01','2016-01-07')
        return early_close_days
