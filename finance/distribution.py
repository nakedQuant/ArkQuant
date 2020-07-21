# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod


class CapitalAssign(ABC):
    """
        distribution base class
    """
    def __init__(self):
        self.adjust_portal = AdjustedArray()

    @abstractmethod
    def compute(self,asset_types,cash):
        raise NotImplementedError


class Average(CapitalAssign):

    def compute(self,assets,cash):
        per_cash = cash / len(assets)
        return {event.sid : per_cash for event in assets}


class Turtle(CapitalAssign):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例
        --- 收益率的sharp_ratio
    """
    def __init__(self,window):

        self._window = window

    def handle_data(self,sids,dt):
        adjust_arrays = self.adjust_portal.load_pricing_adjusted_array(
                                                            dt,
                                                            self.window,
                                                            ['open','high','low','close','volume'],
                                                            sids)
        return adjust_arrays

    def compute(self,dt,assets,cash):
        """基于数据的波动性以及均值"""
        raise NotImplementedError


class Kelly(CapitalAssign):
    """
        基于策略的胜率反向推导出凯利仓位
        ---- 策略胜率
    """
    def __init__(self,hitrate):
        assert hitrate , 'strategy hirate must not be None'
        self.win_rate = hitrate

    def _calculate_kelly(self,sid):
        """
            标的基于什么策略生产的
        """
        rate = self.win_rate[sid.reason]
        return 2 * rate -1

    def compute(self,dt,assets,cash):
        kelly_weight = {
                        asset: cash * self._calculate_kelly(asset)
                        for asset in assets
                        }
        return kelly_weight