# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class BaseFuse(object):

    @classmethod
    def protect(cls):
        raw = input('Y or N:')
        if raw.upper() == 'Y':
            print('to be continued')
        elif raw.upper() == 'N':
            exit(0)
        else:
            raise ValueError


class Fuse(BaseFuse):
    """
        当持仓组合在一定时间均值低于threshold --- 继续执行还是退出
    """
    def __init__(self,
                 fuse=0.85,
                 window=5):
        self.fuse = fuse
        self.window = window

    def trigger(self, portfolio):
        net_value = portfolio.portfolio_daily_value
        if net_value[-self.window:].mean() / net_value[0] <= self.fuse:
            super().protect()


__all__ = ['Fuse']
