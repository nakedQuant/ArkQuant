# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC


class ExecutionStyle(ABC):
    """Base class for order execution styles.
    """

    @abstractmethod
    def get_limit_price(self,_is_buy):
        """
        Get the limit price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_price(self,_is_buy):
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

    def get_limit_price(self,_is_buy):
        return None

    def get_stop_price(self,_is_buy):
        return None


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
        check_stoplimit_prices(limit_price, 'limit')

        self.limit_price = limit_price

    def get_limit_price(self,_is_buy):
        return self.limit_price

    def get_stop_price(self,_is_buy):
        return None


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
        check_stoplimit_prices(stop_price, 'stop')

        self.stop_price = stop_price

    def get_limit_price(self,_is_buy):
        return None

    def get_stop_price(self, _is_buy):
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
        check_stoplimit_prices(limit_price, 'limit')
        check_stoplimit_prices(stop_price, 'stop')

        self.limit_price = limit_price
        self.stop_price = stop_price

    def get_limit_price(self, _is_buy):
        return self.limit_price,

    def get_stop_price(self, _is_buy):
        return self.stop_price,

def check_stoplimit_prices(price, label):
    """
    Check to make sure the stop/limit prices are reasonable and raise
    a BadOrderParameters exception if not.
    """
    try:
        if not isfinite(price):
            raise Exception(
                "Attempted to place an order with a {} price "
                    "of {}.".format(label, price)
            )
    # This catches arbitrary objects
    except TypeError:
        raise Exception(
            "Attempted to place an order with a {} price "
                "of {}.".format(label, type(price))
        )

    if price < 0:
        raise Exception(
            "Can't place a {} order with a negative price.".format(label)
        )