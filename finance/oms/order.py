# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from enum import Enum
import uuid


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
    FOK =  'fok'
    FAK =  'fak'


class BaseOrder(object):

    def __init__(self, asset, price, ticker, style, slippage_model):
        self.asset = asset
        self._price = price
        self._created_dt = ticker
        self.style = style
        self.slippage_model = slippage_model

    def make_id(self):
        return uuid.uuid4().hex()

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    def fit_slippage(self):
        point = 1 + self.slippage_model.calculate_slippage_factor()
        self.price = self._price * point

    def check_trigger(self,preclose):
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
        # 设定价格限制 , iterator里面的对象为第一个为price
        self.fit_slippage()
        bottom = preclose * (1 - self._style.get_stop_price)
        upper = preclose * (1 + self._style.get_limit_price)
        if bottom <= self.price <= upper:
            return True
        return False

    def __eq__(self, other):
        if isinstance(other, Order) and self.__dict__ == other.__dict__:
            return True
        return False

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_dict(self):
        dct = {name: getattr(self.name)
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


class Order(BaseOrder):
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
    __slot__ = ['asset','amount','price','_created_dt','style','slippage_model']

    def __init__(self, asset, amount, price, ticker, style, slippage_model):
        self.amount = amount
        # --- 找到baseOrder 父类 -- self
        super(BaseOrder, self).__init__(self, asset, price, ticker, style, slippage_model)
        self.broker_order_id = self.make_id()


class PriceOrder(BaseOrder):
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
    __slot__ = ['asset','capital','price','_created_dt','style','slippage_model']

    def __init__(self,asset, capital, price, ticker, style, slippage_model):
        self.capital = capital
        super(BaseOrder, self).__init__(self, asset, price, ticker, style, slippage_model)
        self.broker_order_id = self.make_id()


__all__ = [Order,PriceOrder]