# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
import warnings


def _deprecated_getitem_method(name, attrs):
    """Create a deprecated ``__getitem__`` method that tells users to use
    getattr instead.

    Parameters
    ----------
    name : str
        The name of the object in the warning message.
    attrs : iterable[str]
        The set of allowed attributes.

    Returns
    -------
    __getitem__ : callable[any, str]
        The ``__getitem__`` method to put in the class dict.
    """
    attrs = frozenset(attrs)
    msg = (
        "'{name}[{attr!r}]' is deprecated, please use"
        " '{name}.{attr}' instead"
    )

    def __getitem__(self, key):
        """``__getitem__`` is deprecated, please use attribute access instead.
        """
        warnings(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
        if key in attrs:
            return getattr(self, key)
        raise KeyError(key)

    return __getitem__


class Portfolio(object):
    """Object providing read-only access to current portfolio state.

    Parameters
    ----------
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.

    positions : zipline.protocol.Positions
        Dict-like object containing information about currently-held positions.

    """
    __slots__ = ['start_cash', 'portfolio_value', '_cash_flow',
                 'pnl', 'returns', 'utility', 'positions']

    def __init__(self, capital_base=0.0):
        self.portfolio_value = capital_base
        self.positions_values = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.utility = 0.0
        self.positions = None
        self._cash_flow = 0.0
        self.start_cash = capital_base - self.cash_flow

    @property
    def cash_flow(self):
        return self._cash_flow

    @cash_flow.setter
    def cash_flow(self, capital):
        return capital

    def __getattr__(self, item):
        return self.__slots__[item]

    def __repr__(self):
        return "Portfolio({0})".format(self.__slots__)

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
            return position_values / self.portfolio_value
