# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, pandas as pd
from gateway.driver.data_portal import portal
from finance.order import Order


class Generator(object):
    """
        transfer short to long on specific pipeline
    """
    def __init__(self,
                 delay,
                 blotter,
                 division_model):
        self.delay = delay
        self.blotter = blotter
        self.division_model = division_model

    def yield_capital(self, asset, capital, portfolio, dts):
        capital_orders = self.division_model.divided_by_capital(asset, capital, portfolio, dts)
        print('generator capital_orders', capital_orders)
        capital_transactions = self.blotter.create_bulk_transactions(capital_orders, dts)
        print('capital_transactions', capital_transactions)
        return capital_transactions

    def yield_position(self, position, portfolio, dts):
        holding_orders = self.division_model.divided_by_position(position, portfolio, dts)
        holding_transactions = self.blotter.create_bulk_transactions(holding_orders, dts)
        return holding_transactions

    def yield_interactive(self, position, asset, portfolio, dts):
        """
            p -- position , c --- event , dts --- pd.Timestamp or str
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
        # 卖出持仓
        short_transactions = self.yield_position(position, portfolio, dts)
        short_prices = np.array([txn.price for txn in short_transactions])
        short_amount = np.array([txn.amount for txn in short_transactions])
        # 切换之间存在时间差，默认以minutes为单位
        tickers = [pd.Timedelta(minutes='%dminutes' % self.delay) + txn.created_dt for txn in short_transactions]
        tickers = [ticker for ticker in tickers if ticker.hour < 15]
        # 根据ticker价格比值
        minutes = portal.get_spot_value(dts, asset, 'minute', ['close'])
        ticker_prices = np.array([minutes[ticker] for ticker in tickers])
        # 模拟买入订单数量
        ratio = short_prices[:len(tickers)] / ticker_prices
        ticker_amount = ratio * short_amount[:len(tickers)]
        # 生成对应的买入订单
        orders = [Order(asset, *args) for args in zip(ticker_prices, ticker_amount, tickers)]
        long_transactions = self.blotter.create_bulk_transactions(orders, dts)
        return short_transactions, long_transactions


__all__ = ['Generator']
