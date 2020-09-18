# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from _calendar.trading_calendar import calendar

__all__ = ['Freq']


class Freq(object):
    """
        every_day week_start week_end month_start month_end (specific trading day)
        # grouped_by_sid = source_df.groupby(["sid"])
        # group_names = grouped_by_sid.groups.keys()
        # group_dict = {}
        # for group_name in group_names:
        #     group_dict[group_name] = grouped_by_sid.get_group(group_name)
        # for col_name in df.columns.difference(['sid'])
    """
    def __init__(self):
        self.sessions = [pd.Timestamp(s) for s in calendar.all_sessions]

    def minute_rules(self, kwargs):
        """
        :return:specific ticker , e,g --- 9:30,10:30
        """
        minutes = [session + pd.Timedelta(hours=kwargs['hour'], minutes=kwargs['minute']) for session in self.sessions]
        return minutes

    def week_rules(self, td_delta):
        """
        Group by ISO year (0) and week (1) --- isocalendar return iso_year iso_week number  iso week day
        :param td_delta: number
        """
        return set(
            pd.Series(data=self.sessions, index=self.sessions)
            .groupby(lambda x: x.isocalendar()[0:2])
            .nth(td_delta)
            # .astype(np.int64)
        )

    def month_rules(self, td_delta):
        """
        :param td_delta: number
        """
        return set(
            pd.Series(data=self.sessions, index=self.sessions)
            .groupby([lambda x: x.year, lambda x: x.month])
            .nth(td_delta)
            # .astype(np.int64)
        )


if __name__ == '__main__':

    r = Freq()
    print('length', len(r.sessions))
    min = r.minute_rule({'hour': 10, 'minute': 45})
    print('min', len(min), min)
    week = r.week_rules(3)
    print('week', len(week), week)
    mon = r.month_rules(3)
    print('mon', len(mon), mon)
