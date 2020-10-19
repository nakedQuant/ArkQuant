# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from logging import log
from error.errors import ZiplineError
from gateway.driver.data_portal import portal
import warnings


class TradingControlViolation(ZiplineError):
    """
    Raised if an order would violate a constraint set by a TradingControl.
    """
    msg = """
            Order for {capital} or {amount} shares of {asset} at {datetime} violates trading constraint
            {constraint}.
        """.strip()


class TradingControl(ABC):
    """
    Abstract base class representing a fail-safe control on the behavior of any
    algorithm.
    """
    @abstractmethod
    def validate(self,
                 asset,
                 amount,
                 capital,
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
        raise NotImplementedError

    def handle_violation(self,
                         asset,
                         amount,
                         capital,
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
                capital=capital,
                datetime=date_time,
                constraint=constraint)
        elif self.on_error == 'log':
            log.error("Order for {capital} or {amount} shares of {asset} at {dt} "
                      "violates trading constraint {constraint}", capital=capital,
                      amount=amount, asset=asset, dt=date_time,
                      constraint=constraint)
        elif self.on_error == 'warn':
            warnings.warn("Order for {capital} or {amount} shares of {asset} at {dt} "
                          "violates trading constraint {constraint}",capital=capital,
                          amount=amount, asset=asset, dt=date_time,
                          constraint=constraint)
        else:
            raise TradingControlViolation(
                asset=asset,
                amount=amount,
                capital=capital,
                datetime=date_time,
                constraint=constraint)

    def __repr__(self):
        return "{name}({attrs})".format(name=self.__class__.__name__,
                                        attrs=self.__fail_args)


class MaxOrderSize(TradingControl):
    """
    TradingControl representing a limit on the magnitude of any single order
    placed with the given asset.  Can be specified by share or by dollar
    value. 深圳ST股票买入卖出都不受限制，上海买入限制50万股，卖出没有限制
    """

    def __init__(self, kwargs):

        self.length = kwargs['window']
        self.threshold = kwargs.get('threshold', 0.05)
        self.on_error = kwargs.get('on_error', 'warn')
        self.__fail_args = 'order amount exceed average asset volume'

    def validate(self,
                 asset,
                 amount,
                 capital,
                 portfolio,
                 algo_datetime):
        """
        Fail if the magnitude of the given order exceeds either self.max_shares
        or self.max_notional.
        """
        volume_window = portal.get_window([asset], algo_datetime, - abs(self.length), ['volume'])
        threshold_volume = volume_window[asset.sid].mean() * self.threshold
        if amount > threshold_volume:
            self.handle_violation(asset, amount, capital, portfolio, algo_datetime)
            control_volume = threshold_volume
            return asset, control_volume, amount
        return asset, amount, capital


class MaxOrderCapital(TradingControl):
    """
    TradingControl representing a limit on the magnitude of any single order
    placed with the given asset.  Can be specified by share or by dollar value
    """
    def __init__(self, kwargs):

        self.length = kwargs['window']
        self.threshold = kwargs.get('threshold', 0.05)
        self.on_error = kwargs.get('on_error', 'warn')
        self.__fail_args = 'order capital exceed average asset amount'

    def validate(self,
                 asset,
                 amount,
                 capital,
                 portfolio,
                 algo_datetime):
        """
        Fail if the magnitude of the given order exceeds either self.max_shares
        or self.max_notional.
        """
        amount_window = portal.get_window([asset], algo_datetime, - abs(self.length), ['amount'])
        threshold_amount = amount_window[asset.sid].mean() * self.threshold
        if capital > threshold_amount:
            self.handle_violation(asset, capital, portfolio, algo_datetime)
            control_capital = threshold_amount
            return asset, control_capital
        return asset, capital


class MaxPositionValue(TradingControl):
    """
    TradingControl representing a limit on the maximum position size that can
    be held by an algo for a given asset.
    """
    def __init__(self, kwargs):

        self.threshold = kwargs.get('threshold', 0.6)
        assert self.threshold > 0, ValueError("max_notional must be positive.")
        self.on_error = kwargs.get('on_error', 'warn')
        self.__fail_args = 'asset position proportion exceed portfolio limit'

    def validate(self,
                 asset,
                 amount,
                 capital,
                 portfolio,
                 algo_datetime):
        """
        Fail if the given order would cause the magnitude of our position to be
        greater in shares than self.max_shares or greater in dollar value than
        self.max_notional.
        """
        max_holding = portfolio['portfolio_value'] * self.threshold
        position_mappings = {p.sid: p.last_sale_price * p.amount for p in portfolio['positions']}
        holding_value = position_mappings.get(asset.sid, 0.0)
        # 已有持仓 不同的策略生成相同的标的 --- sid
        if capital + holding_value > max_holding:
            self.handle_violation(asset, capital, portfolio, algo_datetime)
            available_capital = max_holding - holding_value
            return asset, available_capital
        return asset, capital


class LongOnly(TradingControl):
    """
    TradingControl representing a prohibition against holding short positions.
    """
    def __init__(self, on_error='fail'):
        self.on_error = on_error
        self.__fail_args = 'short action is not allowed'

    def validate(self,
                 asset,
                 amount,
                 capital,
                 portfolio,
                 algo_datetime):
        """
        Fail if we would hold negative shares of asset after completing this order.
        """
        holdings = {p.sid: p.amount for p in portfolio.positions}
        holding = holdings[asset.sid] + amount
        assert holding >= 0, self.handle_violation(asset, amount, capital, portfolio, algo_datetime)
        return asset, amount, capital


class UnionControl(TradingControl):

    def __init__(self, controls):
        self.controls = controls if isinstance(controls, list) else [controls]

    def validate(self,
                 asset,
                 amount,
                 capital,
                 portfolio,
                 algo_datetime):

        for control in self.controls:
            asset, amount, capital = control.validate(asset, amount, capital, portfolio, algo_datetime)
        return asset, amount, capital


__all__ = ['MaxOrderSize', 'MaxOrderCapital', 'MaxPositionValue', 'LongOnly', 'UnionControl']
