# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from finance.order import Order


class Transaction(object):

    __slots__ = ['asset', 'amount', 'price', 'created_dt', 'cost']

    def __init__(self,
                 asset=None,
                 amount=None,
                 price=None,
                 dts=None,
                 cost=None):
        self.asset = asset
        self.amount = amount
        self.price = price
        self.created_dt = dts
        self.cost = cost

    def __repr__(self):
        template = (
            "{cls}(asset={asset},amount={amount},"
            "created_dt={created_dt},price={price},cost={cost})"
        )
        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            amount=self.amount,
            price=self.price,
            created_dt=self.created_dt,
            cost=self.cost
        )

    def __getitem__(self, attr):
        return self.__slots__[attr]

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
        print('cost', cost)
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
