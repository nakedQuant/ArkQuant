# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC
import numpy as np


class SlippageModel(ABC):

    @abstractmethod
    def calculate_slippage_factor(self, *args):
        raise NotImplementedError


class NoSlippage(SlippageModel):
    """
        ideal model
    """

    def calculate_slippage_factor(self):
        return 1.0


class FixedBasisPointSlippage(SlippageModel):
    """
        basics_points * 0.0001
    """

    def __init__(self, basis_points=1.0):
        super(FixedBasisPointSlippage, self).__init__()
        self.basis_points = basis_points

    def calculate_slippage_factor(self, *args):
        return self.basis_points


class MarketImpact(SlippageModel):
    """
        基于成交量进行对市场的影响进行测算
    """
    def __init__(self,func = np.exp):
        self.adjust_func = func

    def calculate_slippage_factor(self,target,volume):
        psi = target / volume.mean()
        factor = self.adjust_func(psi)
        return factor
