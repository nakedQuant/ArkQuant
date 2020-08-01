# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from finance.oms.order import  Order,PriceOrder
from ._protocol import MutableView


class Transaction(object):

    __slots__ = ['asset','amount','price','_created_dt']

    def __init__(self, asset,amount,price,dt):
        self_ = MutableView(self)
        self_.asset = asset
        self_.amount = amount
        self_.price = price
        self_.created_dt = dt

    def __repr__(self):
        template = (
            "{cls}(asset={asset}, dt={dt},"
            " amount={amount}, price={price})"
        )
        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            dt=self.dt,
            amount=self.amount,
            price=self.price
        )

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setattr__(self, key, value):
        raise NotImplementedError('immuatable object')

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name : self.name for name in self.__slots__}
        return p_dict


def create_transaction(order,fee):
    """
    :param order: Ticker order or Price order
    :param minutes: minutes bar
    :param commission: commission
    :return: transaction
    """
    asset = order.asset
    sign = -1 if order.direction == 'negative' else 1
    if isinstance(order, PriceOrder):
        amount = np.floor(order.capital * (1 - fee) / (order.price * asset.tick_size))
        assert amount  < asset.tick_size , ("Transaction magnitude must be at least 100 share")
    elif isinstance(order,Order):
        amount = order.amount
    else:
        raise TypeError('unkown order type')
    transaction = Transaction(
        asset = asset,
        amount = sign * amount,
        price = order.price,
        created_dt = order._created_dt
    )
    return transaction
