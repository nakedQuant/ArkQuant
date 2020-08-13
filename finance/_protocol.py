# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from gateWay.asset.assets import Asset
from finance.portfolio import _deprecated_getitem_method


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
    __slots__ = ['_mutable_view_obj']

    def __init__(self, ob):
        object.__setattr__(self, '_mutable_view_ob', ob)

    def __getattr__(self, item):
        return getattr(self._mutable_view_ob, item)

    def __setattr__(self, attr, value):
        # vars() 函数返回对象object的属性和属性值的字典对象 --- 扩展属性类型 ,不改变原来的对象属性
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._mutable_view_ob)


class Position(object):
    """
    A protocol position which is not mutated ,but inner can be changed
    """
    __slots__ = ['_underlying_position']

    def __init__(self, inner):
        self._underlying_position = MutableView(inner)

    def __getattr__(self, attr):
        # return self.__dict__[attr]
        return self._underlying_position[attr]

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
    __slots__ = ['settled_cash', 'total_value', 'position_values', 'positions', 'pnl']

    def __init__(self, portfolio):
        self.settled_cash = portfolio.start_cash
        self.total_value = portfolio.portfolio_value
        self.position_values = portfolio.position_values
        self.positions = portfolio.positions
        self.pnl = portfolio.pnl
        # leverage = np.inf

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Portfolio objects')

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
