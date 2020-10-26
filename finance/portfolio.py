# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from toolz import valmap, keymap, merge_with
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
    __slots__ = ['start_cash', 'loan', 'pnl', 'returns', '_cash_flow', 'utility', 'positions',
                 'portfolio_value', 'positions_values', 'portfolio_daily_value']

    def __init__(self, sim_params):
        capital_base = sim_params.capital_base
        self.loan = sim_params.loan
        self.pnl = 0.0
        self.returns = 0.0
        self.utility = 0.0
        self._cash_flow = 0.0
        # positions --- mappings
        self.positions = None
        self.positions_values = 0.0
        self.portfolio_value = capital_base
        self.start_cash = capital_base - self._cash_flow
        self.portfolio_daily_value = pd.Series(capital_base, dtype='float64')

    @property
    def cash_flow(self):
        return self._cash_flow

    @cash_flow.setter
    def cash_flow(self, val):
        self._cash_flow = val

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
            'utility'
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
            # due to asset varies from tag name --- different pipelines has the same sid
            p_values = valmap(lambda x:  x.last_sale_price * x.amount, self.positions)
            p_values = keymap(lambda x: x.sid, p_values)
            aggregate = merge_with(sum, p_values)
            weights = pd.Series(aggregate) / self.portfolio_value
        else:
            weights = pd.Series(dtype='float')
        return weights

    def to_dict(self):
        return self.__dict__


__all__ = ['Portfolio']
