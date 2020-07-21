# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from enum import Enum
from abc import ABC , abstractmethod
import uuid , numpy as np
import math

class StyleType(Enum):
    """
        Market Price (市价单）
    """
    LMT = 'lmt'
    BOC = 'boc'
    BOP = 'bop'
    ITC = 'itc'
    B5TC = 'b5tc'
    B5TL = 'b5tl'
    FOK =  'fok'
    FAK =  'fak'


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
        if not np.isfinite(price):
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





class Order(ABC):

    def make_id(self):
        return  uuid.uuid4().hex()

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    @property
    def status(self):
        self._status = OrderStatus.OPEN

    @status.setter
    def status(self,status):
        self._status = status

    def to_dict(self):
        dct = {name : getattr(self.name)
               for name in self.__slots__}
        return dct

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%s)" % self.to_dict().__repr__()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__()

    @abstractmethod
    def check_trigger(self,price,dt):
        """
        Given an order and a trade event, return a tuple of
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.

        Orders that have been triggered already (price targets reached),
        the order's current values are returned.
        """
        raise NotImplementedError


class TickerOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出

    """
    __slot__ = ['asset','_created_dt','capital']

    def __init__(self,asset,ticker,capital):
        self.asset = asset
        self._created_dt = ticker
        self.order_capital = capital
        self.direction = math.copysign(1,capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self,dts):
        if dts >= self._created_dt:
            return True
        return False


class RealtimeOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
        实时订单
    """
    __slot__ = ['asset', 'capital']

    def __init__(self, asset, capital):
        self.asset = asset
        self.order_capital = capital
        self.direction = math.copysign(1, capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self, dts):
        return True


class PriceOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        限价单 --- 执行买入算法， 如果确定标的可以买入，偏向于涨的概率，主动买入而不是被动买入

        买1 价格超过卖1，买方以卖1价成交
    """
    __slot__ = ['asset','amount','lmt']

    def __init__(self,asset,amount,price):
        self.asset = asset
        self.amount = amount
        self.lmt_price = price
        self._created_dt = dt
        self.direction = math.copysign(1,self.amount)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.LMT

    def check_trigger(self,bid):
        if bid <= self.lmt_price:
            return True
        return False
