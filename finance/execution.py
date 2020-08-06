# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np


class ExecutionStyle(ABC):
    """Base class for order execution styles.
    """

    @abstractmethod
    def get_limit_price_ratio(self, _is_buy):
        """
        Get the limit price ratio for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_price_ratio(self, _is_buy):
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

    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price_ratio(self, _is_buy):
        return -np.inf

    def get_stop_price_ratio(self, _is_buy):
        return np.inf


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
    def __init__(self, limit_price):
        self.limit_price = limit_price

    def get_limit_price_ratio(self, _is_buy):
        return self.limit_price

    def get_stop_price_ratio(self, _is_buy):
        return -np.inf


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
    def __init__(self, stop_price):
        self.stop_price = stop_price

    def get_limit_price_ratio(self, _is_buy):
        return np.inf

    def get_stop_price_ratio(self, _is_buy):
        return self.get_stop_price()


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
    def __init__(self, limit_price, stop_price):
        self.limit_price = limit_price
        self.stop_price = stop_price

    def get_limit_price_ratio(self, _is_buy):
        return self.limit_price,

    def get_stop_price_ratio(self, _is_buy):
        return self.stop_price,

