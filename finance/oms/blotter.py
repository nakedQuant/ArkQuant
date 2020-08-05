# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np , pandas as pd
from finance.transaction import create_transaction


class Blotter(object):

    def __init__(self,
                 creator,
                 commission,
                 multiplier = 1,
                 delay = 1):
        self._order_creator = creator
        self.commission = commission
        self.multiplier = multiplier
        self.delay = delay

    @staticmethod
    def create_bulk_transactions(orders,fee):
        txns = [create_transaction(order,fee) for order in orders]
        return txns

    def simulate_capital_txn(self,asset,capital,dts):
        """
            基于capital --- 买入标的
        """
        orders = self._order_creator.simulate_capital_order(asset,capital,dts)
        #将orders --- transactions
        fee = self.commission.calculate_rate(asset,'positive',dts)
        txns = self.create_bulk_transactions(orders,fee)
        #计算效率
        efficiency = sum([order.per_capital for order in orders]) / capital
        return txns,efficiency

    def simulate_txn(self,asset,amount,dts,direction):
        sizes,data = self._order_creator.calculate_size_arrays(asset,amount,dts)
        direct_orders = self._order_creator.simulate_order(asset,sizes,data,direction)
        p_fee = self.commission.calculate_rate(asset,direction,dts)
        transactions = self.create_bulk_transactions(direct_orders,p_fee)
        #计算效率
        uility = sum([txn.amount for txn in transactions]) / amount
        return transactions , uility

    def simulate_dual_txn(self,p,c,dts):
        """
            holding , asset ,dts
            基于触发器构建 通道 基于策略 卖出 --- 买入
            principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
            订单 --- priceOrder TickerOrder Intime
            engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
            dual -- True 双方向
                  -- False 单方向（提交订单）
            eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
                  --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
            具体逻辑：
                当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
                保证一次卖出成交金额可以覆盖买入标的
            优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
            成交的订单放入队列里面，不断的get
            针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
            订单优先级 --- Intime (first) > TickerOrder > priceOrder
            基于asset计算订单成交比例
            获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
            卖出标的 --- 对应买入标的 ，闲于的资金
        """
        # #卖出持仓
        p_transactions,p_uility = self.simulate_txn(p.asset,p.amount,dts,'negative')
        #计算效率
        # 执行对应的买入算法
        c_data = self._order_creator._create_data(dts,c)
        # 切换之间存在时间差，默认以minutes为单位
        c_tickers = [pd.Timedelta(minutes='%dminutes'%self.delay) + txn.created_dt for txn in p_transactions]
        #根据ticker价格比值
        c_ticker_prices = np.array([c_data[ticker] for ticker in c_tickers])
        p_transaction_prices = np.array([p_txn.price for p_txn in p_transactions])
        ratio = p_transaction_prices / c_ticker_prices
        # 模拟买入订单数量
        c_sizes = [np.floor(p.amount * ratio) for p in p_transactions]
        #生成对应的买入订单
        c_orders = self._order_creator.simulate_order(c, c_sizes,c_data,'positive')
        #订单 --- 交易
        c_fee = self.commission_rate(c,'positive',dts)
        c_transactions = self.create_bulk_transactions(c_orders,c_fee)
        #计算效率
        c_uility = sum([txn.amount for txn in c_transactions]) / sum(c_sizes)
        return p_transactions,p_uility,c_transactions,c_uility