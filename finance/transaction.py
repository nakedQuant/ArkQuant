# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

class Transaction(object):

    # @expect_types(asset=Asset)
    __slots__ = ['asset_type','amount','price','dt']

    def __init__(self, asset,amount,price,dts):
        self.asset = asset
        self.amount = amount
        self.price = price
        self._created_dt = dts

    def __getitem__(self, name):
        return self.__dict__[name]

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

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name : self.name for name in self.__slots__}
        return p_dict


def create_transaction(price,amount,asset,dt):

    # floor the amount to protect against non-whole number orders
    # TODO: Investigate whether we can add a robust check in blotter
    # and/or tradesimulation, as well.
    amount_magnitude = int(abs(amount))

    if amount_magnitude < 100:
        raise Exception("Transaction magnitude must be at least 100 share")

    transaction = Transaction(
        amount=int(amount),
        price=price,
        asset = asset,
        dt = dt
    )

    return transaction