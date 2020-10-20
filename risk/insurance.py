# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np
from _calendar.trading_calendar import calendar


class Fuse(object):
    """
        当持仓组合在一定时间均值低于threshold，则执行manual exit
    """
    def __init__(self,
                 thres,
                 window):
        self._thres = thres
        self.length = window

    def trigger(self, signal):
        raise NotImplementedError

    def proc(self, portfolio):
        net_value = portfolio.portfolio_daily_value
        if len(net_value) < self.length:
            return False
        trigger = net_value[-self.length:].mean() / portfolio.base <= self._thres
        self.trigger(trigger)


class CancelPolicy(ABC):
    """
        Abstract cancellation policy interface.
        --- manual interface
    """
    @abstractmethod
    def should_cancel(self, asset):
        """Should open order be cancelled
        Returns
        -------
        should_cancel : bool
        """
        raise NotImplementedError()


class NoCancel(CancelPolicy):
    """Orders are never automatically canceled.
    """

    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, asset):
        return False


class EODCancel(CancelPolicy):
    """
        This policy cancels open orders which created dt in session of last_traded and eod_window
        --- 取消标的退市之前的一段时间的内订单
    """
    def __init__(self, window):
        """
        :param window: int
        """
        self.eod_window = window

    def should_cancel(self, asset):
        last_traded = asset.last_traded
        previous = calendar.dt_window_size(last_traded, self.eod_window)
        return previous <= last_traded.strftime('%Y-%m-%d')


class ExtraCancel(CancelPolicy):
    """
        the policy cancel order which order asset is suffer negative affairs  --- black swat
    """


class ComposedCancel(CancelPolicy):
    """
     compose rules with some composing function
    """
    def __init__(self, policies):

        if not np.all([isinstance(p, CancelPolicy) for p in policies]):
            raise ValueError('only cancel policy can be composed')
        self.sub_policies = policies

    def should_cancel(self, asset):
        return np.all([p.shoud_cancel(asset) for p in self.sub_policies])


