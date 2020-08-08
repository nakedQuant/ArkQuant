# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class Transaction(object):

    __slots__ = ['asset', 'amount', 'price', 'created_dt']

    def __init__(self, asset, amount, price, cost, dt):
        self.asset = asset
        self.amount = amount
        self.price = price
        self.cost = cost
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


def create_transaction(order, commission):
    """
    :param order: Ticker order or Price order
    :param commission: Commission object used for calculating order cost
    :return: transaction
    """
    asset = order.asset
    sign = -1 if order.direction == 'negative' else 1
    # calculate cost
    cost = commission.calculate(order)
    # create txn
    transaction = Transaction(
        asset=asset,
        amount=order.amount * sign,
        price=order.price,
        dt=order.created_dt,
        cost=cost
    )
    return transaction
