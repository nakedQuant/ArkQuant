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


class PriceOrder(object):

    __slots__ = ['asset', 'price', 'amount', 'created_dt']

    def __init__(self,
                 asset,
                 price,
                 amount,
                 ticker=None):
        """
        :param event: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param price: float64
        :param amount: int (restriction)
        :param ticker: price which triggered by ticker
        :param direction: buy or sell
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
        return self.__dict__


class TickerOrder(object):

    __slots__ = ['asset', 'price', 'amount', 'created_dt']

    def __init__(self,
                 asset,
                 amount,
                 ticker,
                 price=None):
        """
        :param event: namedtuple(asset name) --- asset which is triggered by the name of pipeline
        :param price: float64
        :param amount: int (restriction)
        :param ticker: price which triggered by ticker
        :param direction: buy or sell
        :param style: order enum
        :param slippage_model: slippage
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
        return "Order(%r)" % self.to_dict()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__


__all__ = ['PriceOrder', 'TickerOrder']
