# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
from functools import partial
import numpy as np
import pandas as pd
from collections import namedtuple
from abc import ABC, abstractmethod
from finance.oms.order import Order, PriceOrder
from utils.dt_utilty import locate_pos

OrderData = namedtuple('OrderData', 'minutes open_pct pre_close sliding_amount restricted')


class BaseCreated(ABC):

    @abstractmethod
    def yield_tickers_on_size(self, size):
        """
            create
        """
        raise NotImplementedError()

    @abstractmethod
    def yield_size_on_capital(self, asset, dts, capital):
        """
            基于资金限制以及preclose计算隐藏的订单个数
        """
        raise NotImplementedError()

    @abstractmethod
    def simulate_dist(self, *args):
        """
            根据统计分布构建模拟的价格分布用于设立价格订单
            e.g.:
                基于开盘的涨跌幅决定 --- 当天的概率分布
                simulate price distribution to place on transactions
                :param size: number of transactions
                :param raw:  data for compute
                :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
                :return: array of simualtion price
                :return:
        """
        raise NotImplementedError()


class OrderCreated(BaseCreated):
    """
        1 存在价格笼子
        2 无跌停限制但是存在竞价机制（10%基准价格），以及临时停盘制度
        有存在竞价限制，科创板2% ，或者可转债10%
        第十八条 债券现券竞价交易不实行价格涨跌幅限制。
　　             第十九条 债券上市首日开盘集合竞价的有效竞价范围为发行价的上下 30%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%；
        非上市首日开盘集合竞价的有效竞价范围为前收盘价的上下 10%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%。
         一、可转换公司债券竞价交易出现下列情形的，本所可以对其实施盘中临时停牌措施：
    　　（一）盘中成交价较前收盘价首次上涨或下跌达到或超过20%的；
    　　（二）盘中成交价较前收盘价首次上涨或下跌达到或超过30%的。
    """
    def __init__(self,
                 portal,
                 slippage,
                 commission,
                 execution_style,
                 window=1):
        self._portal = portal
        self._slippage_model = slippage
        self._execution_style = execution_style
        self._commission = commission
        self.restricted_window = window
        self._fraction = 0.05

    @property
    def commission(self):
        return self._commission

    @property
    def fraction(self):
        """设立成交量限制，默认为前一个交易日的百分之一"""
        return self._fraction

    @fraction.setter
    def fraction(self, val):
        self._fraction = val

    def _create_data(self, dt, asset):
        """生成OrderData"""
        minutes = self._portal.get_spot_value(asset, dt, 'minute',
                                              ['open', 'high', 'low', 'close', 'volume'])
        open_pct, pre_close = self._portal.get_open_pct(asset, dt)
        windowed_amount = self._portal.get_window_data(asset, dt, self.restricted_window, ['amount'], 'daily')
        restricted = asset.restricted(dt)
        return OrderData(
                        minutes=minutes,
                        open_pct=open_pct,
                        pre_close=pre_close,
                        sliding_amount=windowed_amount[asset.sid],
                        restricted=restricted
                        )

    @staticmethod
    def simulate_dist(data, size):
        """模拟价格分布，以开盘振幅为参数"""
        open_pct = data.open_pct
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if size > 0:
            #模拟价格分布
            dist = 1 + np.copysign(alpha, np.random.beta(alpha, 100, size))
        else:
            dist = [1 + alpha / 100]
        #避免跌停或者涨停
        restricted = data.restricted
        clip_pct = np.clip(dist, (1 - restricted), (1 + restricted))
        sim_prices = clip_pct * data.pre_close
        return sim_prices

    @staticmethod
    def yield_tickers_on_size(size):
        """根据size生成ticker序列"""
        interval = 4 * 60 / size
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        # 确保首尾
        tick_interval = list(chain(*zip(upper, bottom)))[:size - 1]
        tick_interval.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_interval

    def yield_size_on_capital(self, asset, dts, capital):
        """根据capital 生成资金订单"""
        order_data = self._create_data(dts, asset)
        base_capital = self._commission.gen_base_capital(dts)
        # 满足限制
        restricted_capital = order_data.sliding_amount.mean() * self.fraction
        capital = capital if restricted_capital > capital else restricted_capital
        # 确保满足最低的一手
        per_capital = min([asset.tick_size * order_data.pre_close * (1 + asset.restricted), base_capital])
        assert capital < per_capital, ValueError('capital must satisfy the base tick size')
        size = capital / per_capital
        return size, order_data, per_capital

    def _create_price_order(self, asset, size, data, per_capital, direction):
        """
            A股主板，中小板首日涨幅最大为44%而后10%波动，针对不存在价格笼子（科创板，创业板后期对照科创板改革）
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，
            不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的102%，卖出价格不得低于买入价格98%，
            申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        clip_prices = self.simulate_dist(data, size)
        tickers = [locate_pos(price, data.minutes, 'positive') for price in clip_prices]
        orders = [PriceOrder(asset, per_capital, price, ticker,
                             direction, self._execution_style, self._slippage_model)
                  for price, ticker in zip(clip_prices, tickers)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre_close)]
        return trigger_orders

    def _create_ticker_order(self, asset, size, data, per_capital, direction):
        """
            由于价格笼子，科创板可以参考基于时间的设置订单
        """
        ticker_intervals = self.yield_tickers_on_size(size)
        ticker_prices = [data.min[ticker] for ticker in ticker_intervals]
        orders = [PriceOrder(asset, per_capital, price, ticker,
                             direction, self._execution_style, self._slippage_model)
                  for price, ticker in zip(ticker_prices, ticker_prices)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre_close)]
        return trigger_orders

    def simulate_capital_order(self, asset, capital, dts, direction):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        size, order_data, per_capital = self.yield_size_on_capital(asset, dts, capital)
        # 固定参数
        ticker_func = partial(self._create_ticker_order, data=order_data,
                              per_capital=per_capital, direction=direction)
        price_func = partial(self._create_price_order, data=order_data,
                             per_capital=per_capital, direction=direction)
        # 执行订单生成逻辑
        if asset.bid_mechanism:
            trigger_orders = ticker_func(asset=asset, size=size)
        else:
            trigger_orders = price_func(asset=asset, size=size)
        return trigger_orders

    def calculate_size_arrays(self, asset, q, dts):
        """根据目标q --- 生成size序列拆分订单数量"""
        data = self._create_data(dts, asset)
        per_capital = self._commission.gen_base_capital(dts)
        per_size = min([int(per_capital / data.pre_close), q])
        size_array = np.tile([per_size], int(q/per_size))
        size_array[np.random(int(q/per_size))] += q % per_size
        return per_size, data

    def simulate_order(self, asset, amount, dts, direction):
        """
            针对于持仓卖出生成对应的订单 --- amount
            a. 存在竞价机制 --- 通过时点设立ticker_order
            b. 无竞价机制 --- 提前设立具体的固定价格订单 -- 最后收盘的将为成交订单撮合
            c. size_array(默认将订单拆分同样大小) , np.tile([per_size],size)
        """
        size_array, data, per_capital = self.calculate_size_arrays(asset, amount, dts)

        if asset.bid_mechanism:
            tickers = self.yield_tickers_on_size(len(size_array))
            ticker_prices = [data.minute[ticker] for ticker in tickers]
            iterator = zip(ticker_prices, tickers)
        else :
            # simulate_dist 已经包含剔除限制
            simulate_prices = self.simulate_dist(data, len(size_array))
            tickers = [locate_pos(price, data.minutes, direction) for price in simulate_prices]
            iterator = zip(simulate_prices, tickers)
        orders = [Order(asset, amount, *args, self._execution_style,  self._slippage_model) for amount, args in zip(size_array, iterator)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre_close)]
        return trigger_orders
