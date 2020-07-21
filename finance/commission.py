# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod


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
        1、印花税：1‰(卖的时候才收取，此为国家税收，全国统一)。
        2、过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02‰人民币[3])。
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

    def _init_base_cost(self,dt):
        base_fee = 1e-4 if dt > '2015-06-09' else 1e-3
        self.commission_rate = base_fee * self.mulitplier
        self.min_base_cost = self.min_cost / self.commission_rate

    def calculate(self, transaction):
        self._init_base_cost(transaction.dt)
        transaction_cost = transaction.amount * transaction.price
        #stamp_cost 印花税
        stamp_cost = 0 if transaction.amount > 0  else transaction_cost * 1e-3
        transfer_cost = transaction_cost * 1e-5 if transaction.asset.startswith('6') else 0
        trade_cost = transaction_cost * self.commission_rate \
            if transaction_cost > self.min_base_cost else self.min_cost
        txn_cost = stamp_cost + transfer_cost + trade_cost
        return txn_cost