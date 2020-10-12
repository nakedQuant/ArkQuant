# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from enum import Enum
import uuid
from pipe import Event
from gateway.asset.assets import Equity
from finance.slippage import NoSlippage

__all__ = ['Order']


class OrderType(Enum):
    """
        Market Price (市价单）
    """
    LMT = 'lmt'
    BOC = 'boc'
    BOP = 'bop'
    ITC = 'itc'
    B5TC = 'b5tc'
    B5TL = 'b5tl'
    FOK = 'fok'
    FAK = 'fak'


class Order(object):

    __slots__ = ['event', 'price', 'amount', 'created_dt', 'direction', 'execution_style', 'slippage_model']

    def __init__(self, event, price, amount, ticker, direction, style, slippage_model=NoSlippage):
        """
        :param event: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param price: float64
        :param amount: int (restriction)
        :param ticker: price which triggered by ticker
        :param direction: buy or sell
        :param style: order enum
        :param slippage_model: slippage
        """
        self.event = event
        self.price = price
        self.amount = amount
        self.created_dt = ticker
        self.direction = direction
        self.execution_style = style
        self.slippage_model = slippage_model

    @staticmethod
    def make_id():
        return uuid.uuid4().hex()

    @property
    def asset(self):
        return self.event.asset

    @property
    def sid(self):
        return self.asset.sid

    def _fit_slippage(self, alpha):
        point = self.slippage_model.calculate_slippage_factor(alpha)
        self.price = self.price * (point + 1)

    def check_trigger(self, order_data):
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
        --- 可以扩展为bid_mechanism
        """
        pre_close = order_data.pre_close
        # 设定价格限制 , iterator里面的对象为第一个为price
        if self.asset.bid_mechanism:
            # simulate based on tickers
            return True
        else:
            # simulate price to create order and ensure  order price must be available
            bottom = pre_close * (1 - self.execution_style.get_stop_price_ratio())
            upper = pre_close * (1 + self.execution_style.get_limit_price_ratio())
            if bottom <= self.price <= upper:
                # 计算滑价系数
                avg_volume = order_data.window[self.sid]['volume'].mean()
                alpha = self.amount / avg_volume
                self._fit_slippage(alpha)
                return True
            return False

    def __eq__(self, other):
        if isinstance(other, Order) and self.__dict__ == other.__dict__:
            return True
        return False

    def __contains__(self, name):
        return name in self.__dict__

    def to_dict(self):
        dct = {name: getattr(self, name)
               for name in self.__slots__}
        return dct

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%r)" % self.to_dict()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__


if __name__ == '__main__':
    asset = Equity('600000')
    event = Event(asset, 'pipeline')
    price = 9.59
    amount = 10000
    ticker = '2020-10-12 14:42'
    direction = 'buy'
    style = 'lmt'
    order = Order(event, price, amount, ticker, direction, style)
    print('order', order)
