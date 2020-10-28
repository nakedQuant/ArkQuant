# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from functools import lru_cache
from gateway.driver.data_portal import portal


class ExecutionStyle(ABC):
    """
        Base class for order execution styles.
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.
    """
    @staticmethod
    @lru_cache(maxsize=32)
    def get_pre_close(asset, dt):
        open_pct, pre_close = portal.get_open_pct(asset, dt)
        return pre_close

    @abstractmethod
    def get_limit_ratio(self, asset, dts):
        """
        Get the limit price ratio for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_ratio(self, asset, dts):
        """
        Get the stop price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError


class MarketOrder(ExecutionStyle):
    """
    Execution style for orders to be filled at current market price.

    This is the default for orders placed with :func:`~zipline.api.order`.
    """
    def get_limit_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        limit_price = pre_close * (1 + asset.restricted_change(dts))
        return limit_price

    def get_stop_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        stop_price = pre_close * (1 - asset.restricted_change(dts))
        return stop_price


class LimitOrder(ExecutionStyle):
    """
    Execution style for orders to be filled at a price equal to or better than
    a specified limit price.

    Parameters
    ----------
    limit_price : float
        Maximum price for buys, or minimum price for sells, at which the order
        should be filled.
    """
    def __init__(self, limit=0.08):
        self.limit = limit

    def get_limit_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        limit_price = pre_close * (1 + self.limit)
        return limit_price

    def get_stop_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        stop_price = pre_close * (1 - asset.restricted_change(dts))
        return stop_price


class StopOrder(ExecutionStyle):
    """
    Execution style representing a market order to be placed if market price
    reaches a threshold.

    Parameters
    ----------
    stop_price : float
        Price threshold at which the order should be placed. For sells, the
        order will be placed if market price falls below this value. For buys,
        the order will be placed if market price rises above this value.
    """
    def __init__(self, stop=0.07):
        self.stop = stop

    def get_limit_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        limit_price = pre_close * (1 + asset.restricted_change(dts))
        return limit_price

    def get_stop_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        stop_price = pre_close * (1 - self.stop)
        return stop_price


class StopLimitOrder(ExecutionStyle):
    """
    Execution style representing a limit order to be placed if market price
    reaches a threshold.

    Parameters
    ----------
    limit_price : float
        Maximum price for buys, or minimum price for sells, at which the order
        should be filled, if placed.
    stop_price : float
        Price threshold at which the order should be placed. For sells, the
        order will be placed if market price falls below this value. For buys,
        the order will be placed if market price rises above this value.
    """
    def __init__(self, kwargs):
        self.limit = kwargs.get('limit', 0.08)
        self.stop = kwargs.get('stop', 0.07)

    def get_limit_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        limit_price = pre_close * (1 + self.limit)
        return limit_price

    def get_stop_ratio(self, asset, dts):
        pre_close = super().get_pre_close(asset, dts)
        stop_price = pre_close * (1 - self.stop)
        return stop_price


__all__ = [
    'MarketOrder',
    'LimitOrder',
    'StopOrder',
    'StopLimitOrder'
]
