# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC , abstractmethod
from gateWay.assets.assets import Asset
from .position import  Position


class CancelPolicy(ABC):

    def to_asset(self,obj):
        if isinstance(obj, Position):
            asset = Position.inner_position.asset
        elif isinstance(obj, Asset):
            asset = obj
        else:
            raise TypeError()
        return asset

    @abstractmethod
    def should_cancel(self,obj,dt):
        raise NotImplementedError


class RestrictCancel(CancelPolicy):

    def __init__(self):
        self.adjust_array = AdjustArray()

    def should_cancel(self,obj,dts):
        """
            计算当天的open_pct是否达到涨停
            针对买入asset = self.to_asset(obj)
        """


class SwatCancel(CancelPolicy):

    def __init__(self):
        self.black_swat = Bar()

    @classmethod
    def should_cancel(self,obj,dts):
        asset = self.to_asset(obj)
        black = self.black_swat.load_blackSwat_kline(dts)
        try:
            event = black[asset]
        except KeyError:
            event = False
        return event


class ComposedCancel(CancelPolicy):
    """
     compose two rule with some composing function
    """
    def __init__(self,first,second):
        if not np.all(isinstance(first,CancelPolicy) and isinstance(second,CancelPolicy)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second

    def should_trigger(self,order):

        return self.first.should_cancel(order) & self.second.should_cancel(order)