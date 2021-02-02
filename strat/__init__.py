#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from abc import ABC, abstractmethod
from toolz import valfilter


class Signal(ABC):
    """
        strategy composed of strat via pipe framework
    """
    def __init__(self, params):
        self.params = params

    @property
    def final(self):
        # final term --- return sorted assets including priority
        return self.params.get('final', False)

    @abstractmethod
    def _run_signal(self, feed):
        raise NotImplementedError('implement logic of strat')

    def long_signal(self, metadata, mask) -> bool:
        """
        intended for pipeline
        :param mask:  bool
        :param metadata:  metadata which computed by get_loader
        :return: assets
        """
        signals = [self._run_signal(metadata[m.sid]) for m in mask]
        # print('signals', signals)
        zp = valfilter(lambda x: x > self.params.get('threshold', 0), dict(zip(mask, signals)))
        # print('signal mapping', zp)
        if self.final:
            sorted_zp = sorted(zp.items(), key=lambda x: x[1], reverse=True)
            out = [i[0] for i in sorted_zp]
            # print('ordered out', out)
        else:
            out = list(zp.keys())
            # print('ordinary out', out)
        return out

    def short_signal(self, metadata) -> bool:
        """
        intended for ump
        :param metadata: metadata which computed by get_loader
        :return: bool
        """
        val = self._run_signal(metadata)
        return val


__all__ = ['Signal']
