#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from multiprocessing import Pool
from toolz import valmap
from indicator import MA
from signal import Signal


class Cross(Signal):

    def __init__(self, params):
        self.p = params
        self.ma = MA()

    @property
    def final(self):
        # final term --- return sorted assets including priority
        return self.p.get('final', False)

    def _run_signal(self, feed):
        # default -- buy operation
        long = self.ma.compute(feed, {'window': self.p['long']})
        short = self.ma.compute(feed, {'window': self.p['short']})
        deviation = short[-1] - long[-1]
        return deviation

    def long_signal(self, mask, aggdata) -> bool:
        with Pool(processes=len(mask)) as pool:
            signals = [pool.apply_async(self._run_signal, (aggdata[m]))
                       for m in mask]
            zp = valmap(lambda x: x > self.p.get('threshold', 0), dict(zip(mask, signals)))
            if self.final:
                sorted_zp = sorted(zp.items(), key=lambda x: x[1])
                out = [i[0] for i in sorted_zp]
            else:
                out = list(zp.keys())
        return out

    def short_signal(self, feed) -> bool:
        val = self._run_signal(feed)
        return val < 0
