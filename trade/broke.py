# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from multiprocessing import Pool
from functools import partial


class Broker(object):
    """
        combine simpleEngine and blotter module
        engine --- asset --- orders --- transactions
        订单 --- 引擎生成交易 --- 执行计划式的
    """
    def __init__(self,
                 simulation_blotter,
                 allocation_model):
        self.blotter = simulation_blotter
        self.allocation = allocation_model

    def enroll_impl(self, calls, capital, dts):
        """基于资金买入对应仓位"""
        # 资金分配
        risk_manage = self.allocation.compute(calls.values(), capital)
        p_func = partial(self.blotter.simulate,
                         direction='positive',
                         dts=dts)
        # 并行计算
        with Pool(processes=len(calls)) as pool:
            result = [pool.apply_async(p_func,
                                       asset=asset,
                                       capital=risk_manage[asset])
                      for asset in calls.values()]
        # transaction , efficiency
        transactions, utility = list(zip(*result))
        return transactions, utility

    def withdraw_impl(self, puts, dts):
        """单独的卖出仓位"""
        p_func = partial(self.blotter.simulate_txn,
                         direction='negative',
                         dts=dts)
        with Pool(processes=len(puts))as pool:
            results = [pool.apply_async(p_func, p.asset, p.amount)
                       for p in puts]
            # 卖出订单，买入订单分开来
            transactions, utility = zip(*results)
        return transactions, utility

    def interactive_impl(self, duals, dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        """
        p_func = partial(self.blotter.yield_txn, dts=dts)
        with Pool(processes=len(duals))as pool:
            results = [pool.apply_async(p_func, *obj) for obj in duals]
            # 卖出订单，买入订单分开来
            p_transactions, p_utility, c_transactions, c_utility = zip(*results)
        return p_transactions, p_utility, c_transactions, c_utility

    def carry_out(self, engine, ledger):
        """建立执行计划"""
        capital = ledger.porfolio.cash
        negatives, dual, positives, dts = engine.execute_engine(ledger)
        # 直接买入
        positives, pos_utility = self.enroll_impl(positives, capital, dts)
        # 直接卖出
        negatives, neg_utility = self.withdraw_impl(negatives, dts)
        # 卖出 --- 买入
        p_transactions, p_utility, c_transactions, c_utility = self.interactive_impl(dual, dts)
        # 根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        utility_ratio = np.mean(neg_utility + p_utility + c_utility + pos_utility)
        transactions = negatives + p_transactions + c_transactions + positives
        return transactions, utility_ratio
