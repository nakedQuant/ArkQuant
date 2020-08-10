# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import pandas as pd
import warnings
from gateWay.asset.assets import Asset


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
    __slots__ = ['capital_base', 'start_cash', 'portfolio_value', '_cash_flow',
                 'pnl', 'returns', 'utility', 'positions']

    def __init__(self,capital_base=0.0):
        self.capital_base = capital_base
        self.start_cash = capital_base - self.cash_flow
        self.portfolio_value = capital_base
        self._cash_flow = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.utility = 0.0
        self.positions = None

    @property
    def cash_flow(self):
        return self._cash_flow

    @cash_flow.setter
    def cash_flow(self, capital):
        return capital

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
            return position_values / self.portfolio_value


class InnerPosition:
    """The real values of a position.

    This exists to be owned by both a
    :class:`zipline.finance.position.Position` and a
    :class:`zipline.protocol.Position` at the same time without a cycle.
    """
    __slots__ = ['asset', 'amount', 'cost_basis', 'last_sync_price', 'last_sync_date']

    def __init__(self,
                 asset,
                 amount=0,
                 cost_basis=0.0,
                 last_sync_price=0.0,
                 last_sync_date=None):
        self.asset = asset
        self.amount = amount
        self.cost_basis = cost_basis  # per share
        self.last_sync_price = last_sync_price
        self.last_sync_date = last_sync_date

    def __repr__(self):
        return (
                '%s(asset=%r, amount=%r, cost_basis=%r,'
                ' last_sale_price=%r, last_sale_date=%r)' % (
                    type(self).__name__,
                    self.asset,
                    self.amount,
                    self.cost_basis,
                    self.last_sync_price,
                    self.last_sync_date,
                )
        )


class MutableView(object):
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """
    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    __slots__ = ('_mutable_view_obj')

    def __init__(self,ob):
        object.__setattr__(self,'_mutable_view_ob',ob)

    def __getattr__(self, item):
        return getattr(self._mutable_view_ob,item)

    def __setattr__(self,attr,value):
        #vars() 函数返回对象object的属性和属性值的字典对象 --- 扩展属性类型 ,不改变原来的对象属性
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)'%(type(self).__name__,self._mutable_view_ob)


class Position(object):
    """
    A protocol position which is not mutated ,but inner can be changed
    """
    __slots__ = ['_underlying_position']

    def __init__(self,inner):
        self._underlying_position = MutableView(inner)

    def __getattr__(self, attr):
        return self.__dict__[attr]

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Position objects')

    def __repr__(self):
        return 'Position(%r)' % {
            k: getattr(self, k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sale_price',
                'last_sale_date',
            )
        }

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'position', {
            'sid',
            'amount',
            'cost_basis',
            'last_sale_price',
            'last_sale_date',
        },
    )


class Positions(dict):
    """A dict-like object containing the algorithm's current positions.
    """

    def __missing__(self, key):
        if isinstance(key, Asset):
            return Position(InnerPosition(key))
        else:
            raise TypeError("Position lookup expected a value of type Asset but got {0}"
                            " instead.".format(type(key).__name__))


class Account(object):
    """
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    """
    def __init__(self, portfolio):
        self_ = MutableView(self)
        self_.settled_cash = portfolio.cash
        self_.total_value = portfolio.portfolio_value
        self_.cushion = portfolio.cushion
        self_.positions = portfolio.positions
        # leverage = np.inf

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Portfolio objects')

    def __getattr__(self, item):
        return self.__dict__[item]

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'account', {
            'settled_cash',
            'total_value',
            'cushion',
            'positions',
        },
    )
