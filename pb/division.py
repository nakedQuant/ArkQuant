# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from functools import lru_cache
from gateway.driver.data_portal import portal
from finance.order import PriceOrder, TickerOrder
from pb.dist import simple


class BaseDivision(object):

    def __init__(self,
                 slippage,
                 execution_style,
                 distribution=simple):
        self.slippage_model = slippage
        self.execution_style = execution_style
        self.dis = distribution

    @lru_cache(maxsize=32)
    def _init_data(self, asset, dts):
        open_pct, pre_close = portal.get_open_pct(asset, dts)
        return open_pct, pre_close

    def simulate_iterator(self, *args):
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
        raise NotImplementedError()


class CapitalDivision(BaseDivision):
    """
        split order into plenty of tiny orders

        a. calculate amount to determin size
        b. create ticker_array depend on size
        c. simulate order according to ticker_price , ticker_size , ticker_price
            --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
            --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
        d. principle:
            a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
            b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
            c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
        order amount --- negative
    """
    name = 'capital'

    def yield_size_on_capital(self, asset, capital, dts):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        open_pct, pre_close = self._init_data(asset, dts)
        base_capital = pre_close * (1 + asset.restricted) * asset.tick_size
        # size per_size 单位手
        size = np.floor(capital / base_capital)
        per_size = np.ceil(20000 / base_capital)
        """根据目标q --- 生成size序列拆分订单数量"""
        size_array = np.tile([per_size], int(size/per_size))
        size_array[np.random.randint(0, len(size_array), size % per_size)] += 1
        size_array = size_array * asset.tick_size * -1
        # simulate price
        dist = self.dist.simulate_dist(int(size / per_size), open_pct)
        clip_pct = np.clip(dist, (1 - asset.restricted), (1 + asset.restricted))
        sim_prices = clip_pct * pre_close
        return zip(sim_prices, size_array)

    def yield_ticker_on_capital(self, asset, capital, dts):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        open_pct, pre_close = self._init_data(asset, dts)
        base_capital = pre_close * (1 + asset.restricted) * asset.tick_size
        # size per_size 单位手
        size = np.floor(capital / base_capital)
        per_size = np.ceil(20000 / base_capital)
        """根据目标q --- 生成size序列拆分订单数量"""
        size_array = np.tile([per_size], int(size / per_size))
        size_array[np.random.randint(0, len(size_array), size % per_size)] += 1
        size_array = size_array * asset.tick_size * -1
        # simulate ticker
        sim_tickers = self.dist.simulate_ticker(int(size / per_size))
        return zip(sim_tickers, size_array)

    def simulate_iterator(self, asset, capital, dts):
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
        if asset.bid_mechanism:
            iterator = self.yield_ticker_on_capital(asset, capital, dts)
            orders = [TickerOrder(asset, *args) for args in iterator]

        else:
            iterator = self.yield_size_on_capital(asset, capital, dts)
            orders = [PriceOrder(asset, *args) for args in iterator]
        return orders


class PositionDivision(BaseDivision):
    """
        split order into plenty of tiny orders

        a. calculate amount to determin size
        b. create ticker_array depend on size
        c. simulate order according to ticker_price , ticker_size , ticker_price
            --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
            --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
        d. principle:
            a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
            b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
            c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
        order amount --- positive
    """
    name = 'position'

    def yield_size_on_position(self, position, dts):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        asset = position.asset
        open_pct, pre_close = self._init_data(asset, dts)
        base_capital = pre_close * (1 + asset.restricted) * asset.tick_size
        # size per_size 单位手
        per_size = np.ceil(20000 / base_capital) * asset.tick_size
        size = np.floor(position.amount / per_size)
        """根据目标q --- 生成size序列拆分订单数量"""
        size_array = np.tile([per_size], int(size / per_size))
        size_array[np.random.randint(0, len(size_array), size % per_size)] += 1
        # simulate price
        dist = self.dist.simulate_dist(int(size / per_size), open_pct)
        clip_pct = np.clip(dist, (1 - asset.restricted), (1 + asset.restricted))
        sim_prices = clip_pct * pre_close
        return zip(sim_prices, size_array)

    def yield_ticker_on_position(self, position, dts):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        asset = position.asset
        open_pct, pre_close = self._init_data(asset, dts)
        base_capital = pre_close * (1 + asset.restricted) * asset.tick_size
        # size per_size 单位手
        per_size = np.ceil(20000 / base_capital) * asset.tick_size
        size = np.floor(position.amount / per_size)
        """根据目标q --- 生成size序列拆分订单数量"""
        size_array = np.tile([per_size], int(size / per_size))
        size_array[np.random.randint(0, len(size_array), size % per_size)] += 1
        # simulate ticker
        sim_tickers = self.dist.simulate_ticker(int(size / per_size))
        return zip(sim_tickers, size_array)

    def simulate_iterator(self, position, dts):
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
        asset = position.asset
        if asset.bid_mechanism:
            iterator = self.yield_ticker_on_position(position, dts)
            orders = [TickerOrder(asset, *args) for args in iterator]
        else:
            iterator = self.yield_size_on_position(position, dts)
            orders = [PriceOrder(asset, *args) for args in iterator]
        return orders


__all__ = ['CapitalDivision', 'PositionDivision', 'BaseDivision']
