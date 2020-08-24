# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from strategy import Strategy


class Discretizer(Strategy):
    """
        通用模块 close - pre_high
    """

    def __init__(self, params):
        self.params = params

    def _compute(self, data, mask):
        raise NotImplementedError()