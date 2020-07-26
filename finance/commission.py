# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod
import numpy as np


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
    def __init__(self, multiplier = 3):
        self.mulitplier = multiplier

    @property
    def min_cost(self):
        return 5

    @min_cost.setter
    def min_cost(self,val):
        return val

    def gen_base_fee(self,dt):
        base_fee = 1e-4 if dt > '2015-06-09' else 1e-3
        self.commission_rate = base_fee * self.mulitplier
        self.base_capital = self.min_cost / self.commission_rate

    def calculate_rate(self,asset,direction,dts):
        # 印花税 1‰(卖的时候才收取，此为国家税收，全国统一)
        stamp_cost = 0 if direction == 'positive'  else  1e-3
        # 过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02)
        transfer_cost = 2 * 1e-5 if asset.sid.startswith('6') else 0
        # 交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
        commission_cost = self.gen_base_fee(dts)
        #完整的交易费率
        fee = stamp_cost + transfer_cost + commission_cost
        return fee

    def calculate(self, transaction):
        capital = transaction.amount * transaction.price
        direction = 'negative' if np.sign(transaction.amount) == -1 else 'positive'
        fee = self.calculate_rate(transaction.asset,direction,transaction.ticker)
        txn_cost = capital * fee if capital > self.base_capital else self.base_capital * fee
        return txn_cost