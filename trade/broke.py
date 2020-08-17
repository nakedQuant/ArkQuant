# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
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
                 risk_model,
                 allocation_model):
        self.blotter = simulation_blotter
        self.allocation = allocation_model

    def enroll_implement(self, calls, capital, dts):
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
        # transactions, utility = list(zip(*result))
        transactions = chain(*result)
        return transactions

    def withdraw_implement(self, puts, dts):
        """单独的卖出仓位"""
        p_func = partial(self.blotter.simulate_txn,
                         direction='negative',
                         dts=dts)
        with Pool(processes=len(puts))as pool:
            results = [pool.apply_async(p_func, p.asset, p.amount)
                       for p in puts]
            transactions = chain(*results)
        return transactions

    def interactive_implement(self, duals, dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        """
        p_func = partial(self.blotter.yield_txn, dts=dts)
        with Pool(processes=len(duals))as pool:
            results = [pool.apply_async(p_func, *obj) for obj in duals]
            # 卖出订单，买入订单分开来
            p_transactions, c_transactions = zip(*results)
        return p_transactions, c_transactions

    def carry_out(self, engine, ledger):
        """建立执行计划"""
        dts, capital, negatives, dual, positives = engine.execute_algorithm(ledger)
        # 直接买入
        positives = self.enroll_implement(positives, capital, dts)
        # 直接卖出
        negatives = self.withdraw_implement(negatives, dts)
        # 卖出 --- 买入
        p_transactions, c_transactions = self.interactive_implement(dual, dts)
        # monitor --- risk module to left
        # unchanged_positions
        # 根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        # --- 通过portfolio的资金使用效率来度量算法的效率
        # utility_ratio = np.mean(neg_utility + p_utility + c_utility + pos_utility)
        transactions = chain([negatives, positives, p_transactions, c_transactions])
        # 执行成交
        ledger.process_transaction(transactions)
