# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import pandas as pd, pytz, numpy as np
from datetime import datetime
from dateutil import rrule
from toolz import partition_all
from gateWay.driver.db_schema import engine
from ._config import autumn,spring ,Holiday


class TradingCalendar (object):
    """
    元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日 春节 中秋
    """

    def __init__(self):
        self.engine = engine
        self._fixed_holiday()

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

    def dt_window_size(self,dt ,window):
        dt = self._roll_forward(dt,window)
        return dt

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

    def execution_time_from_open(self,sessions):
        opens = [ pd.Timestamp(dt) + 9 * 60 * 60 + 30 * 60 for dt in sessions]
        _opens = [ pd.Timestamp(dt) + 13 * 60 * 60 for dt in sessions]
        # 熔断期间 --- 2次提前收市
        if '20160107' in sessions:
            idx = np.searchsorted('20160107',sessions)
            _opens[idx] = np.nan
        return zip(opens,_opens)

    def excution_time_from_close(self,sessions):
        closes = [ pd.Timestamp(dt) + 11 * 60 * 60 + 30 * 60 for dt in sessions]
        _closes = [ pd.Timestamp(dt) + 15 * 60 * 60 for dt in sessions]
        # 熔断期间 --- 2次提前收市
        if '20160104' in sessions:
            idx = np.searchsorted('20160104',sessions)
            _closes[idx] = pd.Timestamp('20160104') + 13 * 60 * 60 + 34 * 60
        elif '20160107' in sessions:
            idx = np.searchsorted('20160107',sessions)
            closes[idx] = pd.Timestamp('20160107') + 10 * 60 * 60
            _closes[idx] = np.nan
        return zip(closes,_closes)

    def open_and_close_for_session(self,dts):
        # 每天开盘，休盘，开盘，收盘的时间
        opens = self.execution_time_from_open(dts)
        closes = self.excution_time_from_close(dts)
        o_c = zip(opens,closes)
        return o_c

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

    def get_trading_day_near_holiday(self,holiday_name,forward = True):
        if holiday_name not in Holiday:
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
            自1月8日起暂停实施指数熔断机制
            具体：
                2016年1月4日，A股遇到史上首次“熔断”。早盘，两市双双低开，随后沪指一度跳水大跌，跌破3500点与3400点，各大板块纷纷下挫。
                午后，沪深300指数在开盘之后继续下跌，并于13点13分超过5%，引发熔断，三家交易所暂停交易15分钟，恢复交易之后，沪深300指数继续下跌，
                并于13点34分触及7%的关口，三个交易所暂停交易至收市。
                2016年1月7日，早盘9点42分，沪深300指数跌幅扩大至5%，再度触发熔断线，两市将在9点57分恢复交易。开盘后，仅3分钟（10:00），
                沪深300指数再度快速探底，最大跌幅7.21%，二度熔断触及阈值。这是2016年以来的第二次提前收盘，同时也创造了休市最快记录
        """
        early_close_days = self.session_in_range('2016-01-01','2016-01-07')
        return early_close_days


calendar = TradingCalendar()


__all__ = [calendar]