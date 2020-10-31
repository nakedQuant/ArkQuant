# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from gateway.asset.assets import Asset
from util.wrapper import _deprecated_getitem_method


class MutableView(object):
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """
    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    # __slots__ = ['_mutable_view_obj']

    def __init__(self, ob):
        object.__setattr__(self, '_mutable_view_obj', ob)

    def __getattr__(self, item):
        return getattr(self._mutable_view_ob, item)

    def __setattr__(self, attr, value):
        # vars() 函数返回对象object的属性和属性值的字典对象 --- 扩展属性类型 ,不改变原来的对象属性
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._mutable_view_ob)


class InnerPosition(object):

    __slots__ = ['asset', 'name', 'amount', 'cost_basis', 'last_sync_price',
                 'last_sync_date', 'position_returns']

    def __init__(self,
                 asset,
                 amount=0,
                 cost_basis=0.0,
                 last_sync_price=0.0,
                 last_sync_date=None,
                 position_returns=None):
        self.asset = asset
        self.amount = amount
        self.name = asset.tag
        self.cost_basis = cost_basis  # per share
        self.last_sync_price = last_sync_price
        self.last_sync_date = last_sync_date
        self.position_returns = position_returns

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


class Position(object):
    """
    A position held by an algorithm.

    Attributes
    ----------
    asset : zipline.assets.Asset
        The held asset.
    amount : int
        Number of shares held. Short positions are represented with negative
        values.
    cost_basis : float
        Average price at which currently-held shares were acquired.
    last_sale_price : float
        Most recent price for the position.
    last_sale_date : pd.Timestamp
        Datetime at which ``last_sale_price`` was last updated.
    """
    __slots__ = ('_underlying_position',)

    def __init__(self, underlying_position):
        # object.__setattr__(self, '_underlying_position', underlying_position)
        super(Position, self).__setattr__('_underlying_position', underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position, attr)

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Position objects')

    def __repr__(self):
        return 'Position(%r)' % {
            k: getattr(self, k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sync_price',
                'last_sync_date',
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
            'last_sync_price',
            'last_sync_date',
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


__all__ = [
    'MutableView',
    'InnerPosition',
    'Position',
    'Positions',
]
