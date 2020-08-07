# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from finance.oms.order import Order, PriceOrder


class Transaction(object):

    __slots__ = ['asset', 'amount', 'price', 'created_dt']

    def __init__(self, asset, amount, price, dt):
        self.asset = asset
        self.amount = amount
        self.price = price
        self.created_dt = dt

    def __repr__(self):
        template = (
            "{cls}(asset={asset}, dt={dt},"
            " amount={amount}, direction={direction} price={price})"
        )
        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            amount=self.amount,
            direction=self.direction,
            price=self.price,
            dt=self.created_dt,
        )

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setattr__(self, key, value):
        raise NotImplementedError('immutable object')

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name: self.name for name in self.__slots__}
        return p_dict


def create_transaction(order, fee):
    """
    :param order: Ticker order or Price order
    :param fee: float
    :return: transaction
    """
    asset = order.asset
    sign = -1 if order.direction == 'negative' else 1
    if isinstance(order, PriceOrder):
        assert order.price * (1 + fee) * asset.tick_size <= order.capital, \
            "order capital statisfy tick_size after remove cost"
    elif isinstance(order, Order):
        amount = order.amount
    # create txn
    transaction = Transaction(
        asset=asset,
        amount=sign * amount,
        price=order.price,
        dt=order.created_dt
    )
    return transaction
