#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from indicator import EMA
from indicator.technic import MA
from strat import Signal


class Break(Signal):

    name = 'Break'

    def __init__(self, params):
        # p --- window fast slow period
        super(Break, self).__init__(params)
        self.ema = EMA()
        self.ma = MA()

    def _run_signal(self, feed):
        # default -- buy operation
        ema = self.ema.compute(feed, self.params)
        # print('break ema', ema)
        ma = self.ma.compute(feed, self.params)
        # print('break ma', ma)
        deviation = ema[-1] - ma.iloc[-1]
        return deviation

    def long_signal(self, data, mask) -> bool:
        out = super().long_signal(data, mask)
        # print('break out', out)
        return out

    def short_signal(self, feed) -> bool:
        value = super().short_signal(feed) < 0
        return value
