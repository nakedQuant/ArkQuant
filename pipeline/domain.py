# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class Domain(object):
    """
    A domain defines two features:
    1.  Window_length defining the trading_days before the dt ,default one day --- present day
    2. fields which determines the fields of term
    --- 如果创建太多的实例会导致内存过度占用 ，在pipeline执行算法结束后，清楚所有缓存对象 --- 设立定时任务chu
    """
    def __init__(self,fields,window = 1):
        self._fields = fields
        self._window = window

    @property
    def domain_field(self):
        if not self._fields:
            raise ValueError('fields of domain not None')
        else:
            return self._fields

    @property
    def domain_window(self):
        return self._window

    def all_session(self,sdate,edate):
        sessions = self.trading_calendar.session_in_range(sdate,edate)
        return sessions

    def __or__(self,other):
        if isinstance(other,Domain):
            fields = set(self._fields) | set(other._fields)
            max_window = max(self._window,other._window)
            self.domain_field = fields
            self.domain_window = max_window
        else:
            raise Exception('domain type is needed')
        return self