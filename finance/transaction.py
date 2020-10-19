# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from finance.order import Order


class Transaction(object):

    __slots__ = ['asset', 'amount', 'price', 'cost', 'created_dt']

    def __init__(self, asset, amount, price, dts, cost):
        self.asset = asset
        self.amount = amount
        self.price = price
        self.created_dt = dts
        self.cost = cost

    def __repr__(self):
        template = (
            "{cls}(asset={asset}, dt={dt},"
            " amount={amount},created_by={created_by}, price={price}, dt={dt})"
        )
        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            amount=self.amount,
            created_by=self.asset.tag,
            price=self.price,
            dt=self.created_dt,
        )

    def __getitem__(self, attr):
        return self.__dict__[attr]

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name: getattr(self, name) for name in self.__slots__}
        return p_dict


def create_transaction(order, commission):
    """
    :param order: Ticker order or Price order
    :param commission: Commission object used for calculating order cost
    :return: transaction
    """
    if isinstance(order, Order):
        # calculate cost
        cost = commission.calculate(order)
        # create txn
        transaction = Transaction(
            asset=order.asset,
            amount=order.amount,
            price=order.price,
            dts=order.created_dt,
            cost=cost
        )
        return transaction
    else:
        raise ValueError('Order object can transform to transaction')
