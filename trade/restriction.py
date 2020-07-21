# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from abc import ABC,abstractmethod
from functools import reduce
import pandas as pd , numpy as np , operator
from gateWay.assets.assets import Asset


def vectorized_is_element(array, choices):
    """
    Check if each element of ``array`` is in choices.

    Parameters
    ----------
    array : np.ndarray
    choices : object
        Object implementing __contains__.

    Returns
    -------
    was_element : np.ndarray[bool]
        Array indicating whether each element of ``array`` was in ``choices``.
    """
    return np.vectorize(choices.__contains__, otypes=[bool])(array)


class Restrictions(ABC):
    """
    Abstract restricted list interface, representing a set of assets that an
    algorithm is restricted from trading.
    """

    @abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        asset : Asset of iterable of Assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or assets restricted on this dt?

        """
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        """Base implementation for combining two restrictions.
        """
        # If the right side is a _UnionRestrictions, defers to the
        # _UnionRestrictions implementation of `|`, which intelligently
        # flattens restricted lists
        # 调用 _UnionRestrictions 的__or__
        if isinstance(other_restriction, _UnionRestrictions):
            return other_restriction | self
        return _UnionRestrictions([self, other_restriction])


class _UnionRestrictions(Restrictions):
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
        # Filter out NoRestrictions and deal with resulting cases involving
        # one or zero sub_restrictions
        sub_restrictions = [
            r for r in sub_restrictions if not isinstance(r, NoRestrictions)
        ]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]

        new_instance = super(_UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        """
        Overrides the base implementation for combining two restrictions, of
        which the left side is a _UnionRestrictions.
        """
        # Flatten the underlying sub restrictions of _UnionRestrictions
        if isinstance(other_restriction, _UnionRestrictions):
            new_sub_restrictions = \
                self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]

        return _UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return any(
                r.is_restricted(assets, dt) for r in self.sub_restrictions
            )

        return reduce(
            operator.or_,
            (r.is_restricted(assets, dt) for r in self.sub_restrictions)
        )


class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """
    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=False)


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list.
        """
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(
            index=pd.Index(assets),
            # list 内置的__contains__ 方法
            data=vectorized_is_element(assets, self._restricted_set)
        )


class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list.

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, securities_in_list)
        )


class RestrictedBid(Restrictions):
    """
    biding Restrictions based on a security
    """
    def __init__(self):
        self._name = 'bid'

    def is_restricted(self, asset):
        """在临时停牌阶段，投资者可以继续申报也可以撤销申报，并且申报价格不受2%的报价限制。
            复牌时，对已经接受的申报实行集合竞价撮合交易，申报价格最小变动单位为0.01"""
        sid = asset.sid
        bid_limit = 0.02 if sid.startwith('688') else None
        return bid_limit


class RestrictedPrice(Restrictions):
    """
    Pct Restrictions.
    """
    def __init__(self,
                 trading_calendar,
                 length = 5):
        self._calendar = trading_calendar
        self._restricted_window = length
        self._name = 'price'

    def is_restricted(self, asset, dt):
        """
            科创板股票上市后的前5个交易日不设涨跌幅限制，从第六个交易日开始设置20%涨跌幅限制
        """
        sid = asset.sid
        end_dt = self._calendar._roll_forward(dt,self._restricted_window)
        first_traded = asset.first_traded
        if first_traded == dt :
            _limit = np.inf if sid.startwith('688') else 0.44
        elif first_traded <= end_dt:
            _limit = np.inf if self.sid.startwith('688') else 0.1
        else:
            _limit = 0.2 if self.sid.startwith('688') else 0.1
        return _limit


class TemporaryRestriction(object):
    """
        前5个交易日,科创板科创板还设置了临时停牌制度，当盘中股价较开盘价上涨或下跌幅度首次达到30%、60%时，都分别进行一次临时停牌
        单次盘中临时停牌的持续时间为10分钟。每个交易日单涨跌方向只能触发两次临时停牌，最多可以触发四次共计40分钟临时停牌。
        如果跨越14:57则复盘
    """
    def is_restricted(self,asset,dt):
        raise NotImplementedError()


class AfterRestriction(object):
    """
        科创板盘后固定价格交易 15:00 --- 15:30
        若收盘价高于买入申报指令，则申报无效；若收盘价低于卖出申报指令同样无效
        原则 --- 以收盘价为成交价，按照时间优先的原则进行逐笔连续撮合
    """
    def is_restricted(self,asset,dt):
        raise NotImplementedError()
