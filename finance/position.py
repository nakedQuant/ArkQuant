# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, pandas as pd
from finance._protocol import InnerPosition, Position as ProtocolPosition
from _calendar.trading_calendar import calendar


class Position(object):

    __slots__ = ['inner_position', 'position_returns', '_closed']

    def __init__(self,
                 asset,
                 amount=0,
                 cost_basis=0.0,
                 last_sync_price=0.0,
                 last_sync_date=None):

        # 属性只能在inner里面进行更新 --- 变相隔离
        inner = InnerPosition(
                asset=asset,
                amount=amount,
                cost_basis=cost_basis,
                last_sync_price=last_sync_price,
                last_sync_date=last_sync_date,
        )
        self.inner_position = inner
        self.position_returns = pd.Series(index=calendar.all_sessions, dtype='float64')
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def __getattr__(self, item):
        return getattr(self.inner_position, item)

    @property
    # 当天买入仓位不能卖出 --- 需改整个运行逻辑
    def is_freeze(self):
        return False

    @property
    def protocol(self):
        return ProtocolPosition(self.inner_position)

    def handle_split(self, amount_ratio, cash_ratio):
        """
            update the postion by the split ratio and return the fractional share that will be converted into cash (除权）
            零股转为现金 ,重新计算成本,
            散股 -- 转为现金
        """
        adjust_share_count = self.amount(1 + amount_ratio)
        adjust_cost_basics = round(self.cost_basis / amount_ratio, 2)
        scatter_cash = (adjust_share_count - np.floor(adjust_share_count)) * adjust_cost_basics
        left_cash = self.amount * cash_ratio + scatter_cash
        self.inner_position.amount = np.floor(adjust_share_count)
        self.inner_position.cost_basis = adjust_cost_basics
        return left_cash

    def update(self, txn):
        if self.asset == txn.event.asset:
            raise Exception('transaction asset must same with position asset')
        # 持仓基本净值
        base_value = self.amount * self.cost_basis
        # 交易净值 以及成本
        txn_value = txn.amount * txn.price
        txn_cost = txn.cost
        # 根据交易对持仓进行更新
        total_amount = txn.amount + self.amount
        if total_amount < 0:
            raise Exception('put action is not allowed')
        else:
            total_cost = base_value + txn_value + txn_cost
            try:
                self.inner_position.cost_basis = total_cost / total_amount
                self.inner_position.amount = total_amount
            except ZeroDivisionError :
                """ 仓位结清 , 当持仓为0 --- 计算成本用于判断持仓最终是否盈利, _closed为True"""
                self.inner_position.cost_basis = self.cost_basis + txn_cost / txn.amount
                self.inner_position.last_sync_price = txn.price
                self.inner_position.last_sync_date = txn.created_dt
                self._closed = True
            # txn_capital = txn_value + np.copysign(txn_cost, txn_value)
            txn_capital = txn_value - txn_cost
        return txn_capital

    def calculate_returns(self):
        self.position_returns[self.last_sync_date] = self.last_sync_price / self.cost_basis - 1.0

    def __repr__(self):
        template = "asset={asset}," \
                   "amount={amount}," \
                   "cost_basis={cost_basis}," \
                   "last_sync_price={last_sync_price}," \
                   "last_sync_date={last_sync_date}"
        return template.format(
            asset=self.asset,
            amount=self.amount,
            cost_basis=self.cost_basis,
            last_sync_price=self.last_sync_price,
            last_sync_date=self.last_sync_date
        )

    def to_dict(self):
        """
            create a dict representing the state of this position
        """
        return {
            'asset': self.asset,
            'amount': self.amount,
            'origin': self.asset_type.source,
            'cost_basis': self.cost_basis,
            'last_sale_price': self.last_sale_price,
            'created_by': self.name
            }


__all__ = ['Position']
