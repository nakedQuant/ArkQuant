# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from strategy import Strategy
from toolz import valmap, valfilter


class Simple(Strategy):

    def __init__(self, windows):
        self.windows = windows

    def deviation(self, frame):
        short = frame['close'][-min(self.windows):].mean()
        long = frame['close'][-max(self.windows):].mean()
        breakpoint = short > long
        return breakpoint

    def _compute(self, data, mask):
        out = valmap(lambda x: self.deviation(x), data)
        _mask = valfilter(lambda x: x, out)
        return _mask
