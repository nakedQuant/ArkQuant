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
from _calendar import autumn, spring, Holiday
from gateway.driver.api.client import tsclient


class TradingCalendar (object):
    """
    元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日 春节 中秋
    """

    def __init__(self):
        trading_days = tsclient.to_ts_calendar('1990-01-01', '3000-01-01')
        self.all_sessions = trading_days['trade_dt'].values

    @property
    def _fixed_holiday(self):
        non_trading_rules = dict()
        non_trading_rules['spring'] = spring
        non_trading_rules['autumn'] = autumn
        tz = pytz.timezone('Asia/Shanghai')
        start = pd.Timestamp(min(self.all_sessions), tz=tz)
        end = pd.Timestamp(max(self.all_sessions), tz=tz)

        new_year = rrule.rrule(
            rrule.YEARLY,
            byyearday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'new_year': new_year})

        april_4 = rrule.rrule(
            rrule.YEARLY,
            bymonth= 4,
            bymonthday=4,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'tomb': april_4})

        may_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=5,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'labour': may_day})

        national_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=10,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        non_trading_rules.update({'national': national_day})
        return non_trading_rules

    def _roll_forward(self, dt, window):
        """
        Given a date, align it to the _calendar of the pipe's domain.
        dt = pd.Timestamp(dt, tz='UTC')

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
        """
        if window == 0:
            return dt
        pos = self.all_sessions.searchsorted(dt.strftime('%Y-%m-%d'))
        try:
            loc = pos if self.all_sessions[pos] == dt else pos - 1
            forward = self.all_sessions[loc + 1 - window]
            return pd.Timestamp(self.all_sessions[forward])
        except IndexError:
            raise ValueError(
                "Date {} was past the last session for domain {}. "
                "The last session for this domain is {}.".format(
                    dt.date(),
                    self,
                    self.all_sessions[-1].date()
                )
            )

    def dt_window_size(self, dt, window):
        if isinstance(dt, pd.Timestamp):
            before = self._roll_forward(dt, window)
        return before

    def session_in_range(self, start_date, end_date, include):
        """
        :param start_date: pd.Timestamp
        :param end_date: pd.Timestamp
        :param include: bool --- whether include end_date
        :return: sessions
        """
        if end_date < start_date:
            raise ValueError("End date %s cannot precede start date %s." %
                             (end_date.strftime("%Y-%m-%d"),
                              start_date.strftime("%Y-%m-%d")))
        idx_s = np.searchsorted(self.all_sessions, start_date.strftime('%Y-%m-%'))
        idx_e = np.searchsorted(self.all_sessions, end_date.strftime('%Y-%m-%'))
        sessions = self.all_sessions[idx_s, idx_e + 1] if include \
            else self.all_sessions[idx_s, idx_e]
        return sessions

    @staticmethod
    def sessions_in_minutes(dt):
        # return day minutes
        morning_session = pd.date_range(dt + pd.Timedelta(hours=9, minutes=30),
                                        dt+pd.Timedelta(hours=11, minutes=30),
                                        freq='%dminute' % 1)
        after_session = pd.date_range(dt + pd.Timedelta(hours=13, minutes=00),
                                      dt+pd.Timedelta(hours=15, minutes=00),
                                      freq='%dminute' % 1)
        minutes_session = [morning_session] + [after_session]
        return minutes_session

    def session_in_window(self, end_date, window, include):
        """
        :param end_date: '%Y-%m-%d'
        :param window:  int
        :param include: bool --- determin whether include end_date
        :return: sessions
        """
        # assert window != 0, 'sessions means window is not equal with zero'
        if window == 0:
            return [end_date, end_date]
        start_date = self._roll_forward(end_date, window)
        session_labels = self.session_in_range(start_date, end_date, include)
        return session_labels

    @staticmethod
    def execution_time_from_open(sessions):
        opens = [dt + pd.Timedelta(hours=9, minutes=30) for dt in sessions]
        _opens = [dt + pd.Timedelta(hours=13) for dt in sessions]
        # 熔断期间 --- 2次提前收市
        if '20160107' in sessions:
            idx = np.searchsorted('20160107', sessions)
            _opens[idx] = np.nan
        return opens, opens

    @staticmethod
    def execution_time_from_close(sessions):
        closes = [dt + pd.Timedelta(hours=11, minutes=30) for dt in sessions]
        _closes = [dt + pd.Timedelta(hours=15) for dt in sessions]
        # 熔断期间 --- 2次提前收市
        if '20160104' in sessions:
            idx = np.searchsorted('20160104', sessions)
            _closes[idx] = pd.Timestamp('20160104') + pd.Timedelta(hours=13, minutes=34)
        elif '20160107' in sessions:
            idx = np.searchsorted('20160107', sessions)
            closes[idx] = pd.Timestamp('20160107') + pd.Timedelta(hours=10)
            _closes[idx] = np.nan
        return closes, _closes

    def open_and_close_for_session(self, dts):
        # 每天开盘，休盘，开盘，收盘的时间
        open_tuple = self.execution_time_from_open(dts)
        close_tuple = self.excution_time_from_close(dts)
        o_c = zip(open_tuple, close_tuple)
        return o_c

    def compute_range_chunks(self, start_date, end_date, chunk_size):
        """Compute the start and end dates to run a pipe for.

        Parameters
        ----------
        start_date : pd.Timestamp
            The first date in the pipe.
        end_date : pd.Timestamp
            The last date in the pipe.
        chunk_size : int or None
            The size of the chunks to run. Setting this to None returns one chunk.
        """
        sessions = self.session_in_range(start_date, end_date)
        return (
            (r[0], r[-1]) for r in partition_all(chunk_size, sessions)
        )

    def get_trading_day_near_holiday(self, holiday_name, forward=True):
        # forward --- 节日之前 ， 节日之后
        if holiday_name not in Holiday:
            raise ValueError('unidentified holiday name')
        holiday_days = self._fixed_holiday[holiday_name]
        idx_list = [np.searchsorted(self.all_sessions, t) for t in holiday_days]
        if forward:
            trading_list = self.all_sessions[list(map(lambda x: x - 1, idx_list))]
        else:
            trading_list = self.all_sessions[idx_list]
        return trading_list

    @staticmethod
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
        early_close_days = self.session_in_range('2016-01-01', '2016-01-07')
        return early_close_days

# if calendar is None:
#     cal = self.trading_calendar
# elif calendar is calendars.US_EQUITIES:
#     cal = get_calendar('XNYS')
# elif calendar is calendars.US_FUTURES:
#     cal = get_calendar('us_futures')


calendar = TradingCalendar()

__all__ = [calendar]
