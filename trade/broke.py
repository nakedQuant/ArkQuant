# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from multiprocessing import Pool
from functools import partial


class TradingEngine(object):
    """
        --- interactive , orderCreated
        撮合成交
        如果open_pct 达到10% --- 是否买入
        分为不同的模块 创业板，科创板，ETF
        包含 --- sell orders buy orders 同时存在，但是buy_orders --- 基于sell orders 和 ledger
        通过限制买入capital的pct实现分布买入
        但是卖出订单 --- 通过追加未成交订单来实现
        如何连接卖出与买入模块

        由capital --- calculate orders 应该属于在统一模块 ，最后将订单 --- 引擎生成交易 --- 执行计划式的，非手动操作类型的
        剔除ReachCancel --- 10%
        剔除SwatCancel --- 黑天鹅
        原则：
            主要针对于买入标的的
            对于卖出的，遵循最大程度卖出
    """
    def __init__(self,
                 engine,
                 blotter,
                 allocation):
        # simple_engine
        self.engine = engine
        self.blotter = blotter
        self.allocation = allocation

    def implement_based_on_amount(self,puts,dts):
        """单独的卖出仓位"""
        proc = self.blotter.simulate_txn
        p_func = partial(proc,
                         dts = dts,
                         direction = 'negative')
        with Pool(processes=len(puts))as pool:
            results = [ pool.apply_async(p_func,p.asset,p.amount)
                       for p in puts ]
            #卖出订单，买入订单分开来
            txns, uility = zip(*results)
        return txns,uility

    def implemnt_based_on_capital(self,calls,capital,dts):
        """基于资金买入对应仓位"""
        p_func = partial(self.blotter.simulate_capital_txn(dts = dts))
        # 资金分配
        cap_mappings = self.allocation.compute(calls.values(),capital)
        # 并行计算
        with Pool(processes= len(calls)) as pool:
            result = [ pool.apply_async(p_func,
                                       asset = asset,
                                       capital = cap_mappings[asset]
                                       )
                      for asset in calls.values() ]
        # transaction , efficiency
        transactions,uility = list(zip(*result))
        return transactions , uility

    def implement_dual(self,targets,dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        """
        dual_func  = self.blotter.simulate_dual_txn
        p_func = partial(dual_func,dts = dts)
        with Pool(processes=len(targets))as pool:
            results = [ pool.apply_async(p_func,*obj)
                       for obj in targets ]
            #卖出订单，买入订单分开来
            p_txns,p_uility,c_txns,c_uility = zip(*results)
        return p_txns,p_uility,c_txns,c_uility

    def carry_out(self,ledger):
        """建立执行计划"""
        capital, negatives,dual,positives,dts =  self.engine.execute_engine(ledger)
        # 直接卖出
        negatives, neg_uility = self.implement_based_on_amount(negatives,dts)
        # 卖出 --- 买入
        p_txns,p_uility,c_txns,c_uility = self.implement_dual(dual,dts)
        # 直接买入
        positives , pos_uility = self.implemnt_based_on_capital(positives,capital,dts)
        # 根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        uility_ratio = np.mean(neg_uility + p_uility + c_uility + pos_uility)
        transactions = negatives + p_txns + c_txns + positives
        return transactions , uility_ratio