# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod

__all__ = [
    'NoSlippage',
    'FixedBasisPointSlippage',
    'MarketImpact'
]


class SlippageModel(ABC):

    @abstractmethod
    def calculate_slippage_factor(self, *args):
        raise NotImplementedError


class NoSlippage(SlippageModel):
    """
        ideal model
    """

    def calculate_slippage_factor(self):
        return 0.0


class FixedBasisPointSlippage(SlippageModel):
    """
        basics_points * 0.0001
    """

    def __init__(self, basis_points=0.005):
        super(FixedBasisPointSlippage, self).__init__()
        self.basis_points = basis_points

    def calculate_slippage_factor(self):
        return self.basis_points


class MarketImpact(SlippageModel):

    def __init__(self, _func):
        """
        :param _func: to measure market_impact e.g. exp(alpha) - 1
        """
        self.func = _func

    def calculate_slippage_factor(self, alpha):
        """
        :param alpha: float , e.g. amount / volume.mean()
        :return:
        """
        slippage = self.func(alpha)
        return slippage
