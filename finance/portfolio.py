# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from util.wrapper import _deprecated_getitem_method


class Portfolio(object):
    """Object providing read-only access to current portfolio state.

    Parameters
    ----------
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.

    positions : Position object or None
        Dict-like object containing information about currently-held positions.

    """
    __slots__ = ['start_cash', 'portfolio_value', 'positions_values',
                 '_cash_flow', 'pnl', 'returns', 'utility',
                 'positions', 'portfolio_daily_value']

    def __init__(self, capital_base=0.0):
        self.positions = None
        self.positions_values = 0.0
        self.portfolio_value = capital_base
        self.pnl = 0.0
        self.returns = 0.0
        self.utility = 0.0
        self._cash_flow = 0.0
        self.start_cash = capital_base - self._cash_flow
        self.portfolio_daily_value = pd.Series(capital_base, dtype='float64')

    @property
    def cash_flow(self):
        return self._cash_flow

    def daily_value(self, session_ix):
        self.portfolio_daily_value[session_ix] = self.portfolio_value

    def __getattr__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'portfolio', {
            'capital_base',
            'portfolio_value',
            'pnl',
            'returns',
            'cash',
            'positions',
            'uility'
        },
    )

    @property
    def current_portfolio_weights(self):
        """
        Compute each asset's weight in the portfolio by calculating its held
        value divided by the total value of all positions.

        Each equity's value is its price times the number of shares held. Each
        futures contract's value is its unit price times number of shares held
        times the multiplier.
        """
        if self.positions:
            position_values = pd.Series({
                p.sid: (
                        p.last_sale_price *
                        p.amount
                )
                for p in self.positions
            })
            weights = position_values / self.portfolio_value
        else:
            weights = None
        return weights

    def to_dict(self):
        return self.__dict__


__all__ = ['Portfolio']
