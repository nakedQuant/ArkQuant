# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from multiprocessing import Pool
from functools import partial
from finance.control import UnionControl


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
                 trigger,
                 controls,
                 allocation_model):
        self.engine = engine
        self.trigger = trigger
        self.capital_model = allocation_model
        self.trade_controls = UnionControl(controls)

    def set_controls_func(self, portfolio):
        cap_controls = partial(self.trade_controls.validate_call, porfolio=portfolio)
        position_controls = partial(self.trade_controls.validate_put, porfolio=portfolio)
        return cap_controls, position_controls

    def implement_capital(self, positives, capital, algo_control, dts):
        """基于资金买入对应仓位"""
        assets = list(positives.values())
        txn_mappings = dict()
        # controls
        allocation = self.capital_model.compute(assets, capital, dts)
        for asset, cap in allocation.items():
            asset, control_cap = algo_control(asset=asset, capital=cap, algo_datetime=dts)
            txn_mappings[asset] = self.trigger.yield_capital(asset, control_cap, dts)
        return txn_mappings

    def implement_position(self, negatives, algo_control, dts):
        """单独的卖出仓位"""
        txn_mappings = dict()
        for p in negatives.values():
            p = algo_control(p)
            txn_mappings[p.asset] = self.trigger.yield_position(p, dts)
        return txn_mappings

    def implement_duals(self, duals, dts):
        """
            针对一个pipeline算法，卖出 -- 买入
            防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集
        """
        txn_mappings = dict()
        for dual in duals:
            p, asset = dual
            # p.asset 与 asset 不一样
            short, long = self.trigger.yield_interactive(p, asset, dts)
            txn_mappings[p.asset] = short
            txn_mappings[asset] = long
        return txn_mappings

    @staticmethod
    def multi_process(ledger, iterable):
        def proc(dct):
            for k, v in dct.items():
                ledger.process_transaction(v)
        with Pool(processes=3) as pool:
            [pool.apply_async(proc, item) for item in iterable]

    def implement_broke(self, ledger, dts):
        """建立执行计划"""
        capital = ledger.portfolio.start_cash.copy()
        # {pipeline_name : asset} , {pipeline_name : position} , (position, asset)
        positives, negatives, duals = self.engine.execute_algorithm(ledger)
        # init controls
        capital_controls, position_controls = self.set_controls_func(ledger.portfolio)
        # 直接买入
        call_txns = self.implement_capital(positives, capital, capital_controls, dts)
        # 直接卖出
        put_txns = self.implement_position(negatives, position_controls, dts)
        # 卖出 --- 买入
        dual_txns = self.implement_duals(duals, dts)
        # portfolio的资金使用效率评估引擎撮合的的效率 --- 并行执行成交
        self.multi_process(ledger, [call_txns, put_txns, dual_txns])


__all__ = ['Broker']
