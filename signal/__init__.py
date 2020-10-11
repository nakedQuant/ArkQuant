#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
from abc import ABC, abstractmethod

__all__ = ['Signal']


class Signal(ABC):
    """
        strategy composed of signal via pipe framework
    """

    @abstractmethod
    def _run_signal(self, feed):
        raise NotImplementedError('implement logic of signal')

    @abstractmethod
    def long_signal(self, mask, metadata) -> bool:
        raise NotImplementedError('buy signal')

    @abstractmethod
    def short_signal(self, feed) -> bool:
        raise NotImplementedError('sell signal')
