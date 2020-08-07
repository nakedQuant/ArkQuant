# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod
import numpy as np, pandas as pd
from functools import lru_cache


class CommissionModel(ABC):
    """
        交易成本 分为订单如果有交易成本，考虑是否增加额外成本 ； 如果没有交易成本则需要计算
    """
    @abstractmethod
    def calculate(self, transaction):
        raise NotImplementedError


class NoCommission(CommissionModel):

    @staticmethod
    def calculate(order, transaction):
        return 0.0


class Commission(CommissionModel):
    """
        1、印花税：1‰(卖的时候才收取，此为国家税收，全国统一)
        2、过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02‰人民币)
        3、交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
    """
    def __init__(self, multiplier=5):
        self.multiplier = multiplier
        self._commission_rate = None
        self._base_cost = 5

    @property
    def commission_rate(self):
        return self._commission_rate

    @property
    def min_cost(self):
        """为保证满足最小交易成本 --- e.g : 5所需的capital """
        self._base_cost

    @min_cost.setter
    def min_cost(self, val):
        self._base_cost = val

    @lru_cache(maxsize=32)
    def gen_base_capital(self, dt, out=True):
        if isinstance(dt, pd.Timestamp):
            base_rate = 1e-4 if dt > pd.Timestamp('2015-06-09') else 1e-3
        elif isinstance(dt, str):
            base_rate = 1e-4 if dt > '2015-06-09' else 1e-3
        else:
            raise TypeError()
        self._commission_rate = base_rate * self.multiplier
        base_capital = self.min_cost / self._commission_rate
        if out:
            return base_capital

    def _calculate_rate(self, asset, dts, direction):
        # 印花税 1‰(卖的时候才收取，此为国家税收，全国统一)
        stamp_cost = 0 if direction == 'positive' else 1e-3
        # 过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02)
        transfer_cost = 2 * 1e-5 if asset.sid.startswith('6') else 0
        # 交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
        self.gen_base_fee(dts, out=False)
        # 完整的交易费率
        fee = stamp_cost + transfer_cost + self.commission_rate
        return fee

    # def calculate(self, transaction):
    #     """
    #     :param transaction: Transaction object
    #     :return: transaction cost
    #     """
    #     capital = transaction.amount * transaction.price
    #     direction = 'negative' if np.sign(transaction.amount) == -1 else 'positive'
    #     base_capital = self.gen_base_capital(transaction.ticker)
    #     fee = self._calculate_rate(transaction.asset, direction, transaction.ticker)
    #     txn_cost = capital * fee if capital > base_capital else base_capital * fee
    #     return txn_cost

    def calculate(self, order):
        """
        :param order: Order object
        :return: cost for order
        """
        capital = order.amount * order.price
        direction = 'negative' if np.sign(order.amount) == -1 else 'positive'
        base_capital = self.gen_base_capital(order.ticker)
        rate = self._calculate_rate(order.asset, direction, order.ticker)
        txn_cost = capital * rate if capital > base_capital else base_capital * rate
        return txn_cost
