# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
        feature (compute (feed, kwargs)) ; strategy (__init__ (kwargs) compute(feed, mask))
    """

    @abstractmethod
    def _compute(self, data, mask):
        raise NotImplementedError()

    def compute(self, data, mask=None):
        output = self._compute(data, mask)
        return output
