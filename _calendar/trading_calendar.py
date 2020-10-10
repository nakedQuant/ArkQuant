# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, pytz, numpy as np
from weakref import WeakValueDictionary
from datetime import datetime
from dateutil import rrule
from toolz import partition_all
from _calendar import autumn, spring, Holiday
from gateway.driver.client import tsclient

__all__ = ['calendar']


class TradingCalendar (object):
    """
        元旦：1月1日 ; 清明节：4月4日; 劳动节：5月1日; 国庆节:10月1日 春节 中秋
        数据时间格式 %Y-%m-%d (price, splits, rights, ownership, holder, massive, unfreeze)

    """
    cache = WeakValueDictionary()

    def __new__(cls):
        try:
            instance = cls.cache['calendar']
        except KeyError:
            all_sessions = tsclient.to_ts_calendar('1990-01-01', '3000-01-01').values
            # cls.cache['calendar'] = instance = super(TradingCalendar, cls).__new__(cls)._init(all_sessions)
            # 继承方式调用 -- __new__ 方法（实例）
            cls.cache['calendar'] = instance = super().__new__(cls)._init(all_sessions)
        return instance

    def _init(self, trading_days):
        self.all_sessions = trading_days
        return self

    def holiday_sessions(self):
        non_trading_rules = dict()
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
        new_year = [d.strftime('%Y-%m-%d') for d in new_year]
        non_trading_rules.update({'new_year': new_year})

        april_4 = rrule.rrule(
            rrule.YEARLY,
            bymonth= 4,
            bymonthday=4,
            cache=True,
            dtstart=start,
            until=end
        )
        april_4 = [d.strftime('%Y-%m-%d') for d in april_4]
        print('april_4', april_4)
        non_trading_rules.update({'qingming': april_4})

        may_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=5,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        may_day = [d.strftime('%Y-%m-%d') for d in may_day]
        print('may_day', may_day)
        non_trading_rules.update({'labour': may_day})

        national_day = rrule.rrule(
            rrule.YEARLY,
            bymonth=10,
            bymonthday=1,
            cache=True,
            dtstart=start,
            until=end
        )
        national_day = [d.strftime('%Y-%m-%d') for d in national_day]
        print('national_day', national_day)
        non_trading_rules.update({'national': national_day})
        # append
        non_trading_rules['spring'] = spring
        non_trading_rules['autumn'] = autumn
        return non_trading_rules

    def get_trading_day_near_holiday(self, holiday_name, window, forward=True):
        # forward --- 节日之前 ， 节日之后
        if holiday_name not in Holiday:
            raise ValueError('unidentified holiday name')
        holiday_days = self.holiday_sessions[holiday_name]
        idx_list = [np.searchsorted(self.all_sessions, t) for t in holiday_days]
        trading_list = self.all_sessions[list(map(lambda x: x - window if forward else x + window, idx_list))]
        return trading_list

    def _roll_forward(self, dt, window):
        """
        Given a date, align it to the _calendar of the pipe's domain.
        dt = pd.Timestamp(dt, tz='UTC')

        Parameters
        ----------
        dt : str %Y-%m-%d
        window : int negative

        Returns
        -------
        pd.Timestamp
        """
        if window == 0:
            return dt
        if isinstance(dt, pd.Timestamp):
            dt = dt.strftime('%Y-%m-%d')
        pos = self.all_sessions.searchsorted(dt)
        try:
            loc = pos if self.all_sessions[pos] == dt else pos - 1
            forward = self.all_sessions[loc - abs(window) + 1]
            # return pd.Timestamp(self.all_sessions[forward])
            return forward
        except IndexError:
            raise ValueError(
                "Date {} was past the last session for {}. "
                "The last session for this domain is {}.".format(
                    dt,
                    self,
                    self.all_sessions[-1]
                )
            )

    def dt_window_size(self, dt, window):
        pre = self._roll_forward(dt, window)
        return pre

    def session_in_range(self, start_date, end_date):
        """
        :param start_date: pd.Timestamp
        :param end_date: pd.Timestamp
        :return: sessions exclude end_date
        """
        if end_date < start_date:
            raise ValueError("End date %s cannot precede start date %s." %
                             (end_date.strftime("%Y-%m-%d"),
                              start_date.strftime("%Y-%m-%d")))
        idx_s = np.searchsorted(self.all_sessions, start_date)
        idx_e = np.searchsorted(self.all_sessions, end_date)
        sessions = self.all_sessions[idx_s: idx_e]
        return sessions

    def session_in_window(self, end_date, window):
        """
        :param end_date: '%Y-%m-%d'
        :param window:  int
        :return: sessions
        """
        if window == 0:
            return [end_date, end_date]
        start_date = self._roll_forward(end_date, window)
        session_labels = self.session_in_range(start_date, end_date)
        return session_labels

    @staticmethod
    def session_in_minutes(dt):
        dt = dt if isinstance(dt, (pd.Timestamp, datetime)) else pd.Timestamp(dt)
        morning_session = pd.date_range(dt + pd.Timedelta(hours=9, minutes=30),
                                        dt+pd.Timedelta(hours=11, minutes=30),
                                        freq='%dminute' % 1)
        after_session = pd.date_range(dt + pd.Timedelta(hours=13, minutes=00),
                                      dt+pd.Timedelta(hours=15, minutes=00),
                                      freq='%dminute' % 1)
        minutes_session = [morning_session] + [after_session]
        return minutes_session

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

    @staticmethod
    def execution_time_from_open(sessions):
        opens = [pd.Timestamp(dt) + pd.Timedelta(hours=9, minutes=30) for dt in sessions]
        _opens = [pd.Timestamp(dt) + pd.Timedelta(hours=13) for dt in sessions]
        # 熔断期间 --- 2次提前收市
        if '20160107' in sessions:
            idx = np.searchsorted('20160107', sessions)
            _opens[idx] = np.nan
        return opens, opens

    @staticmethod
    def execution_time_from_close(sessions):
        closes = [pd.Timestamp(dt) + pd.Timedelta(hours=11, minutes=30) for dt in sessions]
        _closes = [pd.Timestamp(dt) + pd.Timedelta(hours=15) for dt in sessions]
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


calendar = TradingCalendar()


if __name__ == '__main__':

    days = calendar.holiday_sessions()
    print('days', days)
