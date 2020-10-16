# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from gateway.driver.data_portal import portal


class SlippageModel(ABC):

    @abstractmethod
    def calculate_slippage_factor(self, asset, dts):
        raise NotImplementedError


class NoSlippage(SlippageModel):
    """
        ideal model
    """
    def calculate_slippage_factor(self, asset, dts):
        return 0.0


class FixedBasisPointSlippage(SlippageModel):
    """
        basics_points * 0.0001
    """

    def __init__(self, basis_points=0.005):
        super(FixedBasisPointSlippage, self).__init__()
        self.basis_points = basis_points

    def calculate_slippage_factor(self, asset, dts):
        return self.basis_points


class MarketImpact(SlippageModel):

    def __init__(self,
                 _func,
                 window=10):
        """
        :param _func: to measure market_impact e.g. exp(alpha) - 1
        """
        self.func = _func
        self.length = window

    def calculate_slippage_factor(self, asset, dts):
        """
        :param alpha: float , e.g. pre_volume / volume.mean()
        :return:
        """
        alpha = portal.get_window([asset], dts, self.length, ['amount', 'volume'])
        slippage = self.func(alpha)
        return slippage


__all__ = [
    'NoSlippage',
    'FixedBasisPointSlippage',
    'MarketImpact'
]

