#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from indicator import MA
from strat import Signal


class Cross(Signal):

    name = 'Cross'

    def __init__(self, params):
        super(Cross, self).__init__(params)
        self.ma = MA()

    def _run_signal(self, feed):
        # default -- buy operation
        category = self.params['fields'][0]
        long = self.ma.compute(feed, {'window': max(self.params['window'])})
        print('cross long', long)
        short = self.ma.compute(feed, {'window': min(self.params['window'])})
        print('cross short', short)
        deviation = short[category][-1] - long[category].iloc[-1]
        return deviation

    def long_signal(self, data, mask) -> bool:
        print('params', self.params)
        out = super().long_signal(data, mask)
        return out

    def short_signal(self, feed) -> bool:
        signal = super().short_signal(feed)
        return signal
