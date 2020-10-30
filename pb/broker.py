# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class Broker(object):
    """
        a. calculate amount to determin size
        b. create ticker_array depend on size
        c. simulate order according to ticker_price , ticker_size , ticker_price
            --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
            --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
        d. principle:
            a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
            b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
            c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
        e.
            combine simpleEngine and blotter module
            engine --- asset --- orders --- transactions
            订单 --- 引擎生成交易 --- 执行计划式的
    """
    def __init__(self,
                 engine,
                 generator,
                 allocation_model):
        self.engine = engine
        self.generator = generator
        self.capital_model = allocation_model

    def implement_capital(self, positives, capital, portfolio, dts):
        """基于资金买入对应仓位"""
        # assets = list(positives.values())
        txn_mappings = dict()
        # controls
        # allocation = self.capital_model.compute(assets, capital, dts)
        allocation = self.capital_model.compute(positives, capital, dts)
        print('allocation', allocation)
        for asset, available in allocation.items():
            # print('broker capital', asset, available)
            txn_mappings[asset] = self.generator.yield_capital(asset, available, portfolio, dts)
        # print('txn_mappings', txn_mappings)
        return txn_mappings

    def implement_position(self, negatives, portfolio, dts):
        """单独的卖出仓位"""
        txn_mappings = dict()
        # for p in negatives.values():
        for p in negatives:
            txn_mappings[p.asset] = self.generator.yield_position(p, portfolio, dts)
        return txn_mappings

    def implement_duals(self, duals, portfolio, dts):
        """
            针对一个pipeline算法，卖出 -- 买入
            防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集
        """
        txn_mappings = dict()
        for dual in duals:
            p, asset = dual
            # p.asset 与 asset 不一样
            short, long = self.generator.yield_interactive(p, asset, portfolio, dts)
            txn_mappings[p.asset] = short
            txn_mappings[asset] = long
        return txn_mappings

    @staticmethod
    def multi_broking(ledger, iterable):
        for txn_mappings in iterable:
            for k, v in txn_mappings.items():
                ledger.process_transaction(v)

    def implement_broke(self, ledger, dts):
        """建立执行计划"""
        capital = ledger.portfolio.portfolio_cash
        positives, negatives, duals = self.engine.execute_algorithm(ledger, dts)
        print('broker : initialize engine output', positives, negatives, duals)
        portfolio = ledger.portfolio
        # 直接买入
        call_transactions = self.implement_capital(positives, capital, portfolio, dts)
        print('call_transactions', call_transactions)
        # 直接卖出
        put_transactions = self.implement_position(negatives, portfolio, dts)
        print('put_transactions', put_transactions)
        # 卖出 --- 买入
        dual_transactions = self.implement_duals(duals, portfolio, dts)
        print('dual_transactions', dual_transactions)
        # portfolio的资金使用效率评估引擎撮合的的效率 --- 并行执行成交
        self.multi_broking(ledger, [call_transactions, put_transactions, dual_transactions])


__all__ = ['Broker']
