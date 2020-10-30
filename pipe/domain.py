# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import valfilter

__all__ = ['Domain', 'infer_domain']


class Domain(object):
    """
    A domain defines two features:
    1.  Window_length defining the trading_days before the dt ,default one day --- present day
    2. fields which determines the fields of term
    --- 如果创建太多的实例会导致内存过度占用 ，在pipeline执行算法结束后，清楚所有缓存对象 --- 设立定时任务chu
    """
    def __init__(self, fields, window=1):
        self._fields = fields
        self._window = window

    @property
    def domain_field(self):
        if not self._fields:
            raise ValueError('fields of domain not None')
        else:
            return self._fields

    @domain_field.setter
    def domain_field(self, val):
        self._fields = val

    @property
    def domain_window(self):
        return self._window

    @domain_window.setter
    def domain_window(self, window):
        self._window = window

    def all_session(self, s_date, e_date):
        sessions = self.trading_calendar.session_in_range(s_date, e_date)
        return sessions

    def __or__(self, other):
        if isinstance(other, Domain):
            fields = set(self.domain_field) | set(other.domain_field)
            # print('fields', fields)
            max_window = max(self.domain_window, other.domain_window)
            # print('max_window', max_window)
            self.domain_field = fields
            self.domain_window = max_window
        else:
            raise Exception('domain type is needed')
        return self


def infer_domain(kwargs):
    kw = kwargs.copy()
    # infer domain via params
    domain_fields = kw.pop('fields', ['open', 'high', 'low', 'close', 'amount', 'volume'])
    assert 'window' in kwargs, 'strategy must need window args'
    window = kw.pop('window') if isinstance(kw['window'], int) else max(kw.pop('window'))
    # 所有int类型最大值
    kw_filter = valfilter(lambda x: isinstance(x, int), kw)
    if kw_filter:
        domain_window = max(max(kw_filter.values()), window)
    else:
        domain_window = window
    domain = Domain(domain_fields, domain_window)
    return domain


# if __name__ == '__main__':
#
#     kw = {'window': (5, 10)}
#     domain_1 = Domain(['close'], 2)
#     domain_2 = Domain(['open'], 20)
#     # domain = domain_1 | domain_2
#     # print('domain', domain.domain_window, domain.domain_field)
#     # print('domain_1', domain_1.domain_window, domain_1.domain_field)
#     domain_1 | domain_2
#     print('domain_1', domain_1.domain_window, domain_1.domain_field)
