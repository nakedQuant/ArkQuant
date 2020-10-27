#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from indicator import EMA
from indicator.technic import TEMA
from strat import Signal


class Break(Signal):

    name = 'Break'

    def __init__(self, params):
        # p --- window fast slow period
        super(Break, self).__init__(params)
        self.ema = EMA()
        self.tema = TEMA()

    def _run_signal(self, feed):
        # default -- buy operation
        # category = self.params['fields'][0]
        ema = self.ema.compute(feed, self.params)
        print('break ema', ema)
        tema = self.tema.compute(feed, self.params)
        print('break macd', tema)
        # deviation = ema[category][-1] - tema[category][-1]
        deviation = ema[-1] - tema[-1]
        return deviation

    def long_signal(self, data, mask) -> bool:
        out = super().long_signal(data, mask)
        return out

    def short_signal(self, feed) -> bool:
        value = super().short_signal(feed)
        return value < 0
