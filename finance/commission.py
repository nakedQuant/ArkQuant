# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np, pandas as pd


class CommissionModel(ABC):
    """
        交易成本 分为订单如果有交易成本，考虑是否增加额外成本 ； 如果没有交易成本则需要计算
    """
    @abstractmethod
    def calculate(self, order):
        raise NotImplementedError


class NoCommission(CommissionModel):

    def calculate(self, order):
        return 0.0


class Commission(CommissionModel):
    """
        1、印花税：1‰(卖的时候才收取，此为国家税收，全国统一)
        2、过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02‰人民币)
        3、交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
    """
    def __init__(self, multiplier=5):
        self.multiplier = multiplier
        self.base_cost = 5

    @property
    def min_cost(self):
        """为保证满足最小交易成本 --- e.g : 5所需的capital """
        return self.base_cost

    @min_cost.setter
    def min_cost(self, val):
        self.base_cost = val

    def _generate_fee_rate(self, order):
        dt = order.created_dt
        dt = dt if isinstance(dt, pd.Timestamp) else pd.Timestamp(dt)
        base_rate = 1e-4 if dt > pd.Timestamp('2015-06-09') else 1e-3
        commission_rate = base_rate * self.multiplier
        return commission_rate

    def calculate_rate_fee(self, order):
        asset = order.asset
        # 印花税 1‰(卖的时候才收取，此为国家税收，全国统一)
        stamp_cost = 0 if np.sign(order.amount) == 1 else 1e-3
        # 过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02)
        transfer_cost = 2 * 1e-5 if asset.sid.startswith('6') else 0
        # 交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
        commission_rate = self._generate_fee_rate(order)
        # 完整的交易费率
        fee = stamp_cost + transfer_cost + commission_rate
        return fee

    def calculate(self, order):
        """
        :param order: Order object
        :return: cost for order
        """
        capital = order.amount * order.price
        cost = capital * self.calculate_rate_fee(order)
        cost = cost if cost >= self.min_cost else self.min_cost
        return cost


__all__ = ['NoCommission', 'Commission']
