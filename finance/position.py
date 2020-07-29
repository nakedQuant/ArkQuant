# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from .commission import Commission
from._protocol import InnerPosition ,Position as ProtocolPosition


class Position(object):

    __slots__ = ['inner_position','protocol_position']

    def __init__(self,
                 asset,
                 amount = 0,
                 cost_basis = 0.0,
                 last_sync_price = 0.0,
                 last_sync_date = None,
                 multiplier = 3):

        inner = InnerPosition(
                asset = asset,
                amount = amount,
                cost_basis = cost_basis,
                last_sync_price = last_sync_price,
                last_sync_date = last_sync_date,
        )
        object.__setattr__(self,'inner_position',inner)
        object.__setattr__(self,'protocol_position',ProtocolPosition(inner))
        self.commission = Commission(multiplier)
        self._closed = False

    @property
    def tag(self):
        return self.asset._tag

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    def __getattr__(self, item):
        return getattr(self.inner_position,item)

    def __setattr__(self, key, value):
        setattr(self.inner_position,key,value)

    def handle_split(self,amount_ratio,cash_ratio):
        """
            update the postion by the split ratio and return the fractional share that will be converted into cash (除权）
            零股转为现金 ,重新计算成本,
            散股 -- 转为现金
        """
        adjust_share_count = self.amount(1 + amount_ratio)
        adjust_cost_basics = round(self.cost_basis / amount_ratio,2)
        scatter_cash = (adjust_share_count - np.floor(adjust_share_count)) * adjust_cost_basics
        left_cash = self.amount * cash_ratio + scatter_cash
        self.cost_basis = adjust_share_count
        self.amount = np.floor(adjust_share_count)
        return left_cash

    def update(self,txn):
        """
            原始 --- 300股 价格100
            交易正 --- 100股 价格120 成本 （300 * 100 + 100 *120 ） / （300+100）
            交易负 --- 100股 价格90  成本
            交易负 --- 300股 价格120 成本 300 * 120 * fee
        """
        if self.asset == txn.asset:
            raise Exception('transaction asset must same with position asset')
        capital = txn.amount * txn.price
        total_amount = txn.amount + self.amount
        if total_amount < 0 :
            raise Exception('put action is not allowed in china')
        else:
            txn_cost = self.commission.calculate(txn)
            # 只要产生交易 -- 成本就是增加的
            total_cost = self.amount * self.cost_basis + txn_cost + capital
            try:
                self.cost_basis = total_cost / total_amount
                self.amount = total_amount
            except ZeroDivisionError :
                """ 仓位结清 , 当持仓为0此时cost_basis为交易成本需要剔除 , _closed为True"""
                self.cost_basis = txn.price - self.cost_basis - txn / txn.amount
                self._closed = True
            txn_cash = capital - txn_cost
        return txn_cash

    def __repr__(self):
        template = "asset :{asset} , amount:{amount},cost_basis:{cost_basis}"
        return template.format(
            asset = self.asset,
            amount = self.amount,
            cost_basis = self.cost_basis
        )

    def to_dict(self):
        """
            create a dict representing the state of this position
        :return:
        """
        return {
            'sid':self.asset.sid,
            'amount':self.amount,
            'origin':self.asset_type.source,
            'cost_basis':self.cost_basis,
            'last_sale_price':self.last_sale_price
        }