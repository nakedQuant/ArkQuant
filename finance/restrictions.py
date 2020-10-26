# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from functools import reduce
import pandas as pd, operator
from gateway.asset.assets import Asset
from gateway.asset.finder import init_finder
from _calendar.trading_calendar import calendar


class Restrictions(ABC):
    """
    Abstract restricted list interface, representing a set of asset that an
    algorithm is restricted from trading.
         --- used for pipe which filter asset list
    """

    @abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        assets : Asset of iterable of Assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or asset restricted on this dt?

        """
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        """Base implementation for combining two restrictions.
        """
        # If the right side is a _UnionRestrictions, defers to the
        # _UnionRestrictions implementation of `|`, which intelligently
        # flattens restricted lists
        # 调用 _UnionRestrictions 的__or__
        if isinstance(other_restriction, UnionRestrictions):
            return other_restriction | self
        return UnionRestrictions([self, other_restriction])


class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """
    def is_restricted(self, assets, dt):
        return set(assets)


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of asset
        The asset to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list.
        """
        selector = set(assets) - set(self._restricted_set)
        return selector


class DataBoundsRestrictions(Restrictions):
    """
        a. 剔除上市不足一个月的 --- 次新股波动性太大
    """
    def __init__(self, length=30):
        self.window = length
        self.asset_finder = init_finder()

    def is_restricted(self, assets, dt):
        s_date = calendar.dt_window_size(dt, self.window)
        # a --- asset ipo date and dt excess 30 days
        alive_assets = self.asset_finder.lifetimes([s_date, dt])
        final_assets = set(assets) & set(alive_assets)
        return final_assets


class StatusRestrictions(Restrictions):
    """
        a. 剔除退市的股票
        b. 剔除停盘
    """
    def __init__(self, length=0):
        self.length = length
        self.asset_finder = init_finder()

    def is_restricted(self, assets, dt):
        # a category
        trade_assets = self.asset_finder.can_be_traded(dt)
        # b category
        del_assets = self.asset_finder.delist_assets(dt, self.length)
        final_assets = (set(assets) - set(del_assets)) & set(trade_assets)
        return final_assets


class SwatRestrictions(Restrictions):
    """
        black swat : asset suffers negative affairs
    """
    def is_restricted(self, assets, dt):
        raise NotImplementedError()


class UnionRestrictions(Restrictions):
    """
    A union of a number of sub restrictions.

    Parameters
    ----------
    sub_restrictions : iterable of Restrictions (but not _UnionRestrictions)
        The Restrictions to be added together

    Notes
    -----
    - Consumers should not construct instances of this class directly, but
      instead use the `|` operator to combine restrictions
    """

    def __new__(cls, sub_restrictions):
        sub_restrictions = sub_restrictions if isinstance(sub_restrictions, (list, tuple)) else [sub_restrictions]
        # Filter out NoRestrictions and deal with resulting cases involving
        # one or zero sub_restrictions
        sub_restrictions = [
            r for r in sub_restrictions if not isinstance(r, NoRestrictions)
        ]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]

        new_instance = super(UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        """
        Overrides the base implementation for combining two restrictions, of
        which the left side is a _UnionRestrictions.
        """
        # Flatten the underlying sub restrictions of _UnionRestrictions
        if isinstance(other_restriction, UnionRestrictions):
            new_sub_restrictions = \
                self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]
        return UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return assets if len(set(r.is_restricted(assets, dt)
                                     for r in self.sub_restrictions)) == 1 else None
        return reduce(
            operator.and_,
            (r.is_restricted(assets, dt) for r in self.sub_restrictions)
        )


__all__ = [
    'UnionRestrictions',
    'NoRestrictions',
    'StaticRestrictions',
    'DataBoundsRestrictions',
    'StatusRestrictions'
]
