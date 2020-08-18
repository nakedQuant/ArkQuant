# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from risk.alert import Risk

__all__ = [
    'Manual',
    'PortfolioRisk'
]


class PortfolioRisk(Risk):
    """
        当持仓组合在一定时间均值低于threshold，则执行manual exit
    """
    def __init__(self,
                 window,
                 max_limit,
                 base_capital):
        self.limit = max_limit
        self.measure_window = window
        self.base_capital = base_capital

    def should_trigger(self, portfolio):
        net_value = portfolio.portfolio_daily_value
        net_value.dropna(inplace=True)
        if len(net_value) < self.measure_window:
            return False
        trigger = net_value[-self.measure_window:].mean() / self.base_capital < 1 - self.limit
        return trigger


class Manual(object):
    """
        portfolio --- trigger --- ledger.positions --- broke.withdraw_implement() --- ledger.process_transaction()
        close all positions due to ledger violate manual controls
    """
    def __init__(self, trigger):
        self.trigger = trigger

    def execute_manual_process(self, ledger, broker, dts):
        if self.trigger.should_trigger(ledger.portfolio):
            transactions = broker.withdraw_implement(ledger.positions, dts)
            ledger.process_transaction(transactions)
