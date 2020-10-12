# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from pipe import Event
from gateway.asset.assets import Equity


class Transaction(object):

    __slots__ = ['event', 'amount', 'price', 'created_dt']

    def __init__(self, event, amount, price, cost, dt):
        self.event = event
        self.amount = amount
        self.price = price
        self.cost = cost
        self.created_dt = dt

    def __repr__(self):
        template = (
            "{cls}(asset={asset}, dt={dt},"
            " amount={amount},created_by={created_by}, direction={direction}, price={price}, dt={dt})"
        )
        return template.format(
            cls=type(self).__name__,
            asset=self.event.asset,
            amount=self.amount,
            created_by=self.event.name,
            direction=self.direction,
            price=self.price,
            dt=self.created_dt,
        )

    def __getitem__(self, attr):
        return self.__dict__[attr]

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
    event = order.event
    sign = -1 if order.direction == 'negative' else 1
    # calculate cost
    cost = commission.calculate(order)
    # create txn
    transaction = Transaction(
        event=event,
        amount=order.amount * sign,
        price=order.price,
        dt=order.created_dt,
        cost=cost
    )
    return transaction


if __name__ == '__main__':

    asset = Equity('600000')
    event = Event(asset, 'pipeline')
    amount = 10000
    price = 9.61
    cost = 40
    dt = '2020-10-12 14:59'
    txn = Transaction(event, amount, price, cost, dt)
    print('txn', txn)
