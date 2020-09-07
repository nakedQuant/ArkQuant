# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, pandas as pd

__all__ = ['Resample']


class Resample(object):
    """
        every_day week_start week_end month_start month_end (specific trading day)
        # grouped_by_sid = source_df.groupby(["sid"])
        # group_names = grouped_by_sid.groups.keys()
        # group_dict = {}
        # for group_name in group_names:
        #     group_dict[group_name] = grouped_by_sid.get_group(group_name)
        # for col_name in df.columns.difference(['sid'])
    """
    def __init__(self, session):
        if isinstance(session[0], str):
            self.session = [pd.Timestamp(s) for s in session]
        self.session = session

    def minute_rule(self, kwargs):
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
            pd.Series(data=self.sessions)
            .groupby(self.sessions.map(lambda x: x.isocalendar()[0:2]))
            .nth(td_delta)
            .astype(np.int64)
        )

    def month_rules(self, td_delta):
        """
        :param td_delta: number
        """
        return set(
            pd.Series(data=self.sessions)
            .groupby([self.sessions.year, self.sessions.month])
            .nth(td_delta)
            .astype(np.int64)
        )
