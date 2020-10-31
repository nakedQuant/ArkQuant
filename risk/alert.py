# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np


class Risk(ABC):
    """
        基于对仓位的控制 -- 达到对持仓组合的变相控制
    """
    @abstractmethod
    def should_trigger(self, p):
        raise NotImplementedError()


class NoRisk(Risk):

    def should_trigger(self, position):
        return False


class PositionLossRisk(Risk):

    def __init__(self, risk):
        """
        :param risk : 仓位亏损比例 e.g. 10%
        """
        self._thres = risk

    def should_trigger(self, position):
        returns = position.position_returns
        trigger = returns[-1] <= - abs(self._thres)
        return trigger


class PositionDrawRisk(Risk):
    """
       --- 仓位最大回撤(针对于盈利回撤）
    """
    def __init__(self, withdraw):
        self._thres = withdraw

    def should_trigger(self, position):
        returns = position.position_returns
        top = max(np.cumprod(returns.values()))
        trigger = (returns[-1] - top) / top > self._thres
        return trigger


class UnionRisk(Risk):

    def __init__(self, risk_models):
        self.models = [risk_models] if isinstance(risk_models, Risk) else risk_models

    def should_trigger(self, p):
        trigger = np.any([risk.should_trigger(p) for risk in self.models])
        return trigger


_all__ = [
    'NoRisk',
    'PositionLossRisk',
    'PositionDrawRisk',
    'PortfolioRisk'
]
