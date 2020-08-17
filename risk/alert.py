# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np

_all__ = [
    'PositionLossRisk',
    'PositionDrawRisk',
    'Portfolio'
]


class Risk(ABC):
    """
        基于对仓位的控制 -- 达到对持仓组合的变相控制
    """
    @abstractmethod
    def should_trigger(self, p):
        raise NotImplementedError()


class PositionLossRisk(Risk):

    def __init__(self, risk):
        """
        :param risk : 仓位亏损比例 e.g. 10%
        """
        self._risk = risk

    def should_trigger(self, position):
        returns = position.position_returns.copy()
        returns.dropna(inplace=True)
        trigger = True if returns[-1] < -abs(self._risk) else False
        return trigger


class PositionDrawRisk(Risk):
    """
       --- 仓位最大回撤(针对于盈利回撤）
    """
    def __init__(self, withdraw):
        self.threshold = withdraw

    def should_trigger(self, position):
        returns = position.position_returns.copy()
        returns.dropna(inplace=True)
        top = max(np.cumprod(returns.values()))
        trigger = True if (returns[-1] - top) / top > self.threshold else False
        return trigger


class Portfolio(Risk):
    """
        当持仓组合在一定时间均值低于限制，则提示或者执行manual
    """
    def __init__(self,
                 window,
                 max_limit,
                 base_capital
                 ):
        self.limit = max_limit
        self.measure_window = window
        self.base_capital = base_capital

    def should_trigger(self, portfolio):
        net_value = portfolio.portfolio_daily_value
        net_value.dropna(inplace=True)
        trigger = True if net_value[-self.measure_window:].mean() / self.base_capital < 1 - self.limit else False
        return trigger
