# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod
from gateWay.driver.bar_reader import AssetSessionReader


class CapitalManagement(ABC):
    """
        distribution base class
    """
    @property
    def bundles(self):
        return AssetSessionReader()

    def handle_data(self,sids,dt):
        adjust_arrays = self.bundles.load_raw_arrays(
                                                        dt,
                                                        self.window,
                                                        ['open','high','low','close','volume'],
                                                        sids
                                                    )
        return adjust_arrays

    @abstractmethod
    def compute(self,assets,cash):
        raise NotImplementedError


class Average(CapitalManagement):

    @staticmethod
    def compute(assets,cash):
        per_cash = cash / len(assets)
        return {event.sid : per_cash for event in assets}


class Turtle(CapitalManagement):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例
        --- 收益率的sharp_ratio
    """
    def __init__(self,window):
        self._window = window

    @property
    def window(self):
        return self._window

    def compute(self,dt,assets,cash):
        """基于数据的波动性以及均值"""
        raise NotImplementedError