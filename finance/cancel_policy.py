# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np
from _calendar.trading_calendar import calendar


class CancelPolicy(ABC):
    """
        Abstract cancellation policy interface.
        --- manual interface
    """
    @abstractmethod
    def should_cancel(self, order):
        """Should open order be cancelled
        Returns
        -------
        should_cancel : bool
        """
        raise NotImplementedError()


class NullCancel(CancelPolicy):
    """Orders are never automatically canceled.
    """

    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, order):
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

    def should_cancel(self, order):
        last_traded = order.asset.last_traded
        ticker = order.created_dt
        previous = calendar.dt_window_size(last_traded, self.eod_window)
        return previous <= ticker.strftime('%Y-%m-%d')


class ExtraCancel(CancelPolicy):
    """
        the policy cancel order which order asset is suffer negative affairs  --- black swat
    """
    def __init__(self, root_dir):
        """
        :param root_dir:  black swat securities file
        """
        self.root_dir = root_dir

    def should_cancel(self, order):
        # 保证file update
        black_list = self._load_file()
        return order.asset in black_list


class ComposedCancel(CancelPolicy):
    """
     compose rules with some composing function
    """
    def __init__(self, policies):

        if not np.all([isinstance(p, CancelPolicy) for p in policies]):
            raise ValueError('only StatelessRule can be composed')

        self.sub_policies = policies

    def should_cancel(self, order):

        return np.all([p.shoud_cancel(order) for p in self.sub_policies])


__all__ = ['ComposedCancel', 'EODCancel', 'NullCancel', 'ExtraCancel']
