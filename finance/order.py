# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import uuid
from enum import Enum


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

    __slots__ = ['asset', 'price', 'amount', 'created_dt']

    def __init__(self,
                 asset=None,
                 price=None,
                 amount=None,
                 ticker=None):
        """
        :param event: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param price: float64
        :param amount: int
        :param ticker: price which triggered by ticker
        """
        self.asset = asset
        self.price = price
        self.amount = amount
        self.created_dt = ticker

    @staticmethod
    def make_id():
        return uuid.uuid4().hex

    @property
    def sid(self):
        return self.asset.sid

    def __eq__(self, other):
        if isinstance(other, PriceOrder) and self.__dict__ == other.__dict__:
            return True
        return False

    def __contains__(self, name):
        return name in self.__dict__

    def to_dict(self):
        mappings = {name: getattr(self, name)
                    for name in self.__slots__}
        return mappings

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%r)" % self.to_dict()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__slots__


class PriceOrder(Order):

    __slots__ = ['asset', 'price', 'amount']

    def __init__(self,
                 asset,
                 amount,
                 price):
        """
        :param asset: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param price: float64
        :param amount: int
        """
        super(PriceOrder, self).__init__(asset=asset,
                                         amount=amount,
                                         price=price)

    @staticmethod
    def make_id():
        return uuid.uuid4().hex

    @property
    def sid(self):
        return self.asset.sid

    def __eq__(self, other):
        if isinstance(other, PriceOrder) and self.__dict__ == other.__dict__:
            return True
        return False

    def __contains__(self, name):
        return name in self.__dict__

    def to_dict(self):
        mappings = {name: getattr(self, name)
                    for name in self.__slots__}
        return mappings

    def __repr__(self):
        """
        String representation for this object.
        """
        return "PriceOrder(%r)" % self.to_dict()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__slots__


class TickerOrder(Order):

    __slots__ = ['asset', 'amount', 'created_dt']

    def __init__(self,
                 asset,
                 amount,
                 ticker):
        """
        :param asset: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param amount: int
        :param ticker: price which triggered by ticker
        """
        super(TickerOrder, self).__init__(asset=asset,
                                          amount=amount,
                                          ticker=ticker)

    @staticmethod
    def make_id():
        return uuid.uuid4().hex

    @property
    def sid(self):
        return self.asset.sid

    def __eq__(self, other):
        if isinstance(other, TickerOrder) and self.__dict__ == other.__dict__:
            return True
        return False

    def __contains__(self, name):
        return name in self.__dict__

    def to_dict(self):
        mappings = {name: getattr(self, name)
                    for name in self.__slots__}
        return mappings

    def __repr__(self):
        """
        String representation for this object.
        """
        return "TickerOrder(%r)" % self.to_dict()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__slots__


def transfer_to_order(order, ticker=None, price=None):
    if isinstance(order, PriceOrder) and ticker is not None:
        new_order = Order(asset=order.asset,
                          amount=order.amount,
                          price=order.price,
                          ticker=ticker)
    elif isinstance(order, TickerOrder) and price is not None:
        new_order = Order(asset=order.asset,
                          amount=order.amount,
                          price=price,
                          ticker=order.created_dt)
    else:
        new_order = None
    return new_order


__all__ = ['Order',
           'PriceOrder',
           'TickerOrder',
           'transfer_to_order']


# if __name__ == '__main__':
#
#     import pandas as pd
#     from gateway.asset.assets import Equity
#     equity = Equity('002049')
#     iterables = [(2200.0, 8.08035087719301), (2300.0, 8.08035087719301), (2200.0, 8.08035087719301), (
#                   2200.0, 8.08035087719301), (2300.0, 8.08035087719301)]
#     if equity.bid_mechanism:
#         orders = [TickerOrder(equity, *args) for args in iterables]
#     else:
#         orders = [PriceOrder(equity, *args) for args in iterables]
#     print('c_test orders', orders)
