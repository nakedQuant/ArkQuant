# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from .order import  TickerOrder , PriceOrder


class Transaction(object):

    __slots__ = ['asset','amount','price','_created_dt']

    def __init__(self, asset,amount,price,dt):
        self.asset = asset
        self.amount = amount
        self.price = price
        self._created_dt = dt

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

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name : self.name for name in self.__slots__}
        return p_dict


def create_transaction(asset,size,price,ticker):

    transaction = Transaction(
        amount= int(size),
        price = price,
        asset = asset,
        dt = ticker
    )
    return transaction

def simulate_transaction(order,minutes,commission):
    """
    :param order: Ticker order or Price order
    :param minutes: minutes bar
    :param commission: commission
    :return: transaction
    """
    if isinstance(order, TickerOrder):
        ticker = order.ticker
        txn_price = minutes[ticker]
    elif isinstance(order, PriceOrder):
        assert minutes.min() <= order.price  <= minutes.max(), ValueError('order price out of range')
        txn_price = order.price
        try:
            loc = minutes.values().index(txn_price)
        except ValueError :
            loc = np.searchsorted(minutes.values(),txn_price)
        # 当价格大于卖出价格才会成交，价格低于买入价格才会成交
        ticker = minutes.index[loc] if order.direction == 'negative' else minutes.index[loc - 1]
    else:
        raise ValueError('unkown order type -- %r'%order)
    amount = np.floor(order.capital * ( 1 - commission) / (txn_price * 100))
    if amount  < 100:
        raise Exception("Transaction magnitude must be at least 100 share")

    transaction = Transaction(
        amount=int(amount),
        price = txn_price,
        asset = order.asset,
        dt = ticker
    )
    return transaction
