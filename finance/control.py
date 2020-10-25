# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import warnings, logging, numpy as np
from abc import ABC, abstractmethod
from error.errors import ZiplineError, AccountControlViolation
from gateway.driver.data_portal import portal


class TradingControlViolation(ZiplineError):
    """
    Raised if an order would violate a constraint set by a TradingControl.
    """
    msg = """
            Order for {amount} shares of {asset} at {datetime} violates trading constraint
            {constraint}.
        """.strip()


class TradingControl(object):
    """
    Abstract base class representing a fail-safe control on the behavior of any
    algorithm.
    """
    def __init__(self, on_error='log'):
        self.on_error = on_error

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Before any order is executed by TradingAlgorithm, this method should be
        called *exactly once* on each registered TradingControl object.

        If the specified asset and amount do not violate this TradingControl's
        restraint given the information in `portfolio`, this method should
        return None and have no externally-visible side-effects.

        If the desired order violates this TradingControl's contraint, this
        method should call self.fail(asset, amount).
        """
        return amount

    def handle_violation(self,
                         asset,
                         amount,
                         date_time):
        """
        Handle a TradingControlViolation, either by raising or logging and
        error with information about the failure.

        If dynamic information should be displayed as well, pass it in via
        `metadata`.
        """
        constraint = repr(self)

        if self.on_error == 'fail':
            raise TradingControlViolation(
                asset=asset,
                amount=amount,
                datetime=date_time,
                constraint=constraint)
        elif self.on_error == 'log':
            logging.info("Order for {amount} shares of {asset} at {dt} "
                         "violates trading constraint {constraint}",
                         amount=amount, asset=asset, dt=date_time,
                         constraint=constraint)
        elif self.on_error == 'warn':
            warnings.warn("Order for {amount} shares of {asset} at {dt} "
                          "violates trading constraint {constraint}",
                          amount=amount, asset=asset, dt=date_time,
                          constraint=constraint)
        else:
            raise TradingControlViolation(
                asset=asset,
                amount=amount,
                datetime=date_time,
                constraint=constraint)

    def __repr__(self):
        return "{name}({attrs})".format(name=self.__class__.__name__,
                                        attrs=self.__fail_args)


class MaxOrderSize(TradingControl):
    """
    TradingControl representing a limit on the magnitude of any single order
    placed with the given asset.  Can be specified by share or by dollar value
    """
    def __init__(self,
                 max_notional,
                 sliding_window=1,
                 on_error='log'):
        super(MaxOrderSize, self).__init__(on_error)
        self.max_notional = max_notional
        self.window = sliding_window
        self.on_error = on_error
        self.__fail_args = 'order capital exceed average asset amount'

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Fail if the magnitude of the given order exceeds either self.max_shares
        or self.max_notional.
        """
        amount_window = portal.get_window([asset], algo_datetime, - abs(self.window), ['volume'])
        thres_amount = amount_window[asset.sid].mean() * self.max_notional
        if amount > thres_amount:
            self.handle_violation(asset, amount, algo_datetime)
            amount = thres_amount
        return amount


class MaxPositionSize(TradingControl):
    """
    TradingControl representing a limit on the magnitude of any single order
    placed with the given asset.  Can be specified by share or by dollar
    value. 深圳ST股票买入卖出都不受限制，上海买入限制50万股，卖出没有限制
    """

    def __init__(self,
                 max_notional,
                 sliding_window=1,
                 on_error='log'
                 ):
        super(MaxPositionSize, self).__init__(on_error)
        self.max_notional = max_notional
        self.window = sliding_window
        self.on_error = on_error
        self.__fail_args = 'order amount exceed average asset volume'

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Fail if the magnitude of the given order exceeds either self.max_shares
        or self.max_notional.
        """
        current_share = portfolio.positions[asset].amount
        agg_shares = current_share + amount
        volume_window = portal.get_window([asset], algo_datetime, - abs(self.window), ['volume'])
        max_share = volume_window[asset.sid].mean() * self.max_notional
        too_many_shares = agg_shares > max_share
        if too_many_shares:
            self.handle_violation(asset, amount, algo_datetime)
            amount = max_share
        return amount


class LongOnly(TradingControl):
    """
    TradingControl representing a prohibition against holding short positions.
    """

    def __init__(self):
        super(LongOnly, self).__init__(on_error='fail')

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Fail if we would hold negative shares of asset after completing this
        order.
        """
        if portfolio.positions[asset].amount + amount < 0:
            self.handle_violation(asset, amount, algo_datetime)


class MaxProportion(TradingControl):
    """
    TradingControl representing a limit on the maximum position size that can
    be held by an algo for a given asset.
    """
    def __init__(self,
                 max_notional,
                 on_error='log'):
        super(MaxProportion, self).__init__(on_error)
        self.max_notional = max_notional
        self.on_error = on_error
        self.__fail_args = 'asset position proportion exceed portfolio limit'

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Fail if the given order would cause the magnitude of our position to be
        greater in shares than self.max_shares or greater in dollar value than
        self.max_notional.
        """
        weights = portfolio.current_portfolio_weights

        if weights[asset.sid] >= self.max_notional:
            self.handle_violation(asset, amount, algo_datetime)
            amount = 0
        else:
            max_capital = portfolio.portfolio_value * (self.max_notional - weights)
            price = portfolio.positions[asset].last_sync_price
            multiplier = np.floor(max_capital / (price * asset.tick_size))
            amount = asset.tick_size * multiplier
        return amount


class NoControl(TradingControl):
    """
    TradingControl representing a prohibition against holding short positions.
    """
    def __init__(self):
        super(NoControl, self).__init__()
        self.__fail_args = 'short action is not allowed'

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):
        """
        Fail if we would hold negative shares of asset after completing this order.
        """
        super().validate(asset,
                         amount,
                         portfolio,
                         algo_datetime)


class UnionControl(TradingControl):

    def __init__(self, controls):
        super(UnionControl, self).__init__()
        self.controls = controls if isinstance(controls, list) else [controls]

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime):

        for control in self.controls:
            amount = control.validate(asset, amount, portfolio, algo_datetime)
        return amount

##################
# Account API
##################


class AccountControl(ABC):
    """
    Abstract base class representing a fail-safe control on the behavior of any
    algorithm.
    """

    def __init__(self, **kwargs):
        """
        Track any arguments that should be printed in the error message
        generated by self.fail.
        """
        self.__fail_args = kwargs

    @abstractmethod
    def validate(self,
                 portfolio,
                 account,
                 algo_datetime):
        """
        On each call to handle data by TradingAlgorithm, this method should be
        called *exactly once* on each registered AccountControl object.

        If the check does not violate this AccountControl's restraint given
        the information in `portfolio` and `account`, this method should
        return None and have no externally-visible side-effects.

        If the desired order violates this AccountControl's contraint, this
        method should call self.fail().
        """
        raise NotImplementedError

    def fail(self):
        """
        Raise an AccountControlViolation with information about the failure.
        """
        raise AccountControlViolation(constraint=repr(self))

    def __repr__(self):
        return "{name}({attrs})".format(name=self.__class__.__name__,
                                        attrs=self.__fail_args)


class NetLeverage(AccountControl):
    """
    AccountControl representing a limit on the maximum leverage allowed
    by the algorithm.
    """

    def __init__(self, base_leverage=1.0):
        """
        max_leverage is the gross leverage in decimal form. For example,
        2, limits an algorithm to trading at most double the account value.
        """
        super(NetLeverage, self).__init__(base_leverage=base_leverage)

        if base_leverage is None:
            raise ValueError(
                "Must supply max_leverage"
            )

        if base_leverage < 0:
            raise ValueError(
                "max_leverage must be positive"
            )
        self.base_leverage = base_leverage

    def validate(self,
                 portfolio,
                 account,
                 algo_datetime):
        """
        Fail if the leverage is less than base_leverage means the amount is less than loan
        """
        if account.leverage <= self.base_leverage:
            self.fail()


__all__ = ['MaxPositionSize',
           'MaxOrderSize',
           'MaxProportion',
           'LongOnly',
           'NoControl',
           'UnionControl',
           'NetLeverage']
