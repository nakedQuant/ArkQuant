# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from util.wrapper import _deprecated_getitem_method


class Account(object):
    """
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    """
    __slots__ = ['settled_cash', 'loan', 'total_value',
                 'position_values', 'positions', 'pnl']

    def __init__(self, portfolio):
        self.loan = portfolio.loan
        self.settled_cash = portfolio.start_cash
        self.total_value = portfolio.portfolio_value
        self.position_values = portfolio.position_values
        self.positions = portfolio.positions
        self.pnl = portfolio.pnl

    @property
    def leverage(self):
        leverage = self.total_value / self.loan if self.loan > 0 else np.inf
        return leverage

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    # def __setattr__(self, attr, value):
    #     raise AttributeError('cannot mutate Portfolio objects')

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'account', {
            'settled_cash',
            'total_value',
            'position_values',
            'cushion',
            'positions',
        },
    )
