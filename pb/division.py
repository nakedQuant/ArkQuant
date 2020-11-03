# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np, copy
from gateway.driver.data_portal import portal
from finance.order import PriceOrder, TickerOrder
from finance.control import UnionControl


class Division(object):

    def __init__(self,
                 uncover_model,
                 trading_controls,
                 base_capital=20000):
        self.underneath_func = uncover_model
        self.trade_controls = UnionControl(trading_controls)
        self.base_capital = base_capital

    def _calculate_division_data(self, asset, dts, amount_only=False):
        tick_size = asset.tick_size
        open_change, pre_close = portal.get_open_pct(asset, dts)
        ensure_price = pre_close * (1 + asset.restricted_change(dts))
        # ensure amount at least 1 , base_amount(单位股数）
        # base_amount = max(tick_size, np.ceil(self.base_capital / ensure_price))
        base_amount = tick_size if tick_size * ensure_price >= self.base_capital else \
            np.ceil(self.base_capital / (ensure_price * tick_size)) * tick_size
        per_amount = tick_size * np.floor(base_amount / tick_size) if asset.increment else base_amount
        if amount_only:
            return base_amount
        return open_change, ensure_price, per_amount

    @staticmethod
    def _simulate_iterator(asset, zips):
        """
            针对于持仓卖出生成对应的订单 ， 一般不存在什么限制
            a. 存在竞价机制 --- 通过时点设立ticker_order
            b. 无竞价机制 --- 提前设立具体的固定价格订单 -- 最后收盘的将为成交订单撮合
            c. size_array(默认将订单拆分同样大小) , np.tile([per_size],size)

            A股主板，中小板首日涨幅最大为44%而后10%波动，针对不存在价格笼子（科创板，创业板后期对照科创板改革）
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，
            不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的102%，卖出价格不得低于买入价格98%，
            申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
            由于价格笼子，科创板可以参考基于时间的设置订单
        """
        # if asset.bid_mechanism:
        #     orders = [TickerOrder(asset, *z) for z in zips]
        # else:
        #     orders = [PriceOrder(asset, *z) for z in zips]
        # print('orders', orders)
        orders = [TickerOrder(asset, *z) for z in zips]
        return orders

    def divided_by_capital(self, asset, capital, portfolio, dts):
        """
            split order into plenty of tiny orders
            a. calculate amount to determine size
            b. create ticker_array depend on size
            c. simulate order according to ticker_price , ticker_size , ticker_price
                --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
                --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
            d. principle:
                a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
                b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
                c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
            order amount --- negative

            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合

        """
        open_pct, ensure_price, per_amount = self._calculate_division_data(asset, dts)
        # print('division data', open_pct, ensure_price, per_amount)
        multiplier = capital / (ensure_price * asset.tick_size)
        if multiplier >= 1.0:
            amount = asset.tick_size * np.floor(multiplier) \
                if asset.increment else np.floor(capital / ensure_price)
            # print('ensure amount', amount)
            assert amount >= asset.tick_size, 'amount must be at least tick_size'
            control_amount = self.trade_controls.validate(asset, amount, portfolio, dts)
            # print('capital control_amount', control_amount)
            zip_iterables = self.underneath_func.create_iterables(asset, control_amount, per_amount, dts)
            capital_orders = self._simulate_iterator(asset, zip_iterables)
            # print('capital_orders', capital_orders)
        else:
            capital_orders = []
        return capital_orders

    def divided_by_position(self, position, portfolio, dts):
        # transaction amount 为 负
        asset = position.asset
        amount = - copy.copy(position.amount)
        per_amount = self._calculate_division_data(asset, dts, amount_only=True)
        print('position per amount', per_amount)
        control_amount = self.trade_controls.validate(asset, amount, portfolio, dts)
        print('position control_amount', control_amount)
        iterables = self.underneath_func.create_iterables(asset, control_amount, per_amount, dts)
        position_orders = self._simulate_iterator(asset, iterables)
        print('position_orders', position_orders)
        return position_orders


__all__ = ['Division']
