# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
import numpy as np, pandas as pd
from functools import lru_cache
from abc import ABC, abstractmethod
from finance.order import Order
from broker import OrderData
from finance.cancel_policy import ComposedCancel
from util.dt_utilty import locate_pos

__all__ = ['OrderSimulation']


class BaseSimulation(ABC):
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
    @staticmethod
    @abstractmethod
    def yield_tickers_on_size(size):
        """
            create
        """
        raise NotImplementedError()

    @abstractmethod
    def yield_size_on_capital(self, asset, dts, capital, direction):
        """
            基于资金限制以及preclose计算隐藏的订单个数
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def simulate_dist(*args):
        """
            根据统计分布构建模拟的价格分布用于设立价格订单
            e.g.:
                基于开盘的涨跌幅决定 --- 当天的概率分布
                simulate price distribution to place on transactions
                :param size: number of transactions
                :param raw:  data for compute
                :return: array of nakedquant price
                :return:
        """
        raise NotImplementedError()


class OrderSimulation(BaseSimulation):
    """
        capital or amount --- transform to Order object
        a. calculate amount to determin size
        b. create ticker_array depend on size
        c. simulate order according to ticker_price , ticker_size , ticker_price
            --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
            --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
        d. principle:
            a. 基于信号执行买入 --- 避免分段连续性买入
            b. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
            c. 针对于卖出标的 -- 遵循最大程度卖出（当天）

        logic: asset - capital - controls - orders - check_trigger - execute_cancel_policy (positive)
               asset - amount - orders - check_trigger - execute_cancel_policy (negative)
        执行买入算法的需要涉及比如最大持仓比例，持仓量等限制 ； 而卖出持仓比较简单以最大方式卖出标的
    """
    def __init__(self,
                 data_portal,
                 controls,
                 slippage,
                 commission,
                 cancel_policy,
                 execution_style,
                 window=1):
        self.slippage_model = slippage
        self.data_portal = data_portal
        self.commission_model = commission
        self.execution_style = execution_style
        self.cancel_policy = ComposedCancel(cancel_policy)
        # 限制条件  MaxPositionSize ,MaxOrderSize
        self.max_position_control, self.max_order_control = controls
        # 计算滑价与定义买入capital限制
        self._window = window
        self._fraction = 0.05

    @property
    def fraction(self):
        """设立成交量限制，默认为前一个交易日的百分之一"""
        return self._fraction

    @fraction.setter
    def fraction(self, val):
        self._fraction = val

    def _create_data(self, dt, asset):
        """生成OrderData"""
        minutes = self.data_portal.get_spot_value(asset, dt, 'minute',
                                                  ['open', 'high', 'low', 'close', 'volume'])
        open_pct, pre_close = self.data_portal.get_open_pct(asset, dt)
        window = self.data_portal.get_window_data(asset, dt, self._window, ['amount', 'volume'], 'daily')
        restricted = asset.restricted(dt)
        return OrderData(
                        minutes=minutes,
                        open_pct=open_pct[asset],
                        pre_close=pre_close[asset],
                        window=window,
                        restricted=restricted
                        )

    @staticmethod
    def simulate_dist(data, size):
        """模拟价格分布，以开盘振幅为参数"""
        open_pct = data.open_pct
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if size > 0:
            # 模拟价格分布
            dist = 1 + np.copysign(alpha, np.random.beta(alpha, 100, size))
        else:
            dist = [1 + alpha / 100]
        # 避免跌停或者涨停
        restricted = data.restricted
        clip_pct = np.clip(dist, (1 - restricted), (1 + restricted))
        sim_prices = clip_pct * data.pre_close
        return sim_prices

    @staticmethod
    def yield_tickers_on_size(size):
        """
            a. 根据size生成ticker序列
            b. 14:57时刻最终收盘竞价机制阶段 --- 确保收盘价纳入订单组合
        """
        interval = 4 * 60 / size
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        # 确保首尾
        tick_interval = list(chain(*zip(upper, bottom)))[:size - 1]
        tick_interval.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_interval

    @lru_cache(maxsize=8)
    def yield_size_on_capital(self, asset, dts, capital, direction):
        """根据capital 生成资金订单"""
        order_data = self._create_data(dts, asset)
        base_capital = self.commission_model.gen_base_capital(dts)
        # 满足限制
        restricted_capital = order_data.window[asset.sid]['amount'].mean() * self.fraction
        capital = capital if restricted_capital > capital else restricted_capital
        rate = self.commission_model.calculate_rate(asset, dts, direction)
        # 以涨停价来度量capital
        per_capital = min([asset.tick_size * order_data.pre_close * (1 + asset.restricted), base_capital])
        assert capital < per_capital, ValueError('capital must satisfy the base tick size')
        size = np.floor(capital * (1 - rate) / per_capital)
        return size, order_data

    def calculate_size_arrays(self, asset, q, dts):
        """根据目标q --- 生成size序列拆分订单数量"""
        data = self._create_data(dts, asset)
        per_capital = self.commission_model.gen_base_capital(dts)
        per_size = min([int(per_capital / data.pre_close), q])
        size_array = np.tile([per_size], int(q/per_size))
        size_array[np.random.random(int(q/per_size))] += q % per_size
        return size_array, data

    def _finalize(self, orders, order_data):
        # orders --- trigger_orders --- final_orders --- transactions
        # 计算滑价与触发条件 check_trigger --- slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        trigger_orders = [order for order in orders if order.check_trigger(order_data)]
        # cancel_policy
        final_orders = [odr for odr in trigger_orders if self.cancel_policy.should_cancel(odr)]
        # transactions = [create_transaction(order, self.commission) for order in final_orders]
        return final_orders

    def simulate(self, event, capital, dts, direction, portfolio):
        """
            针对于买入操作
            a. 计算满足最低capital(基于手续费逻辑），同时计算size
            b. 存在竞价机制 --- 基于size设立时点order
            c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
        """
        asset = event.asset
        control_capital = self.max_position_control.validate(asset, None, portfolio, dts)
        size, order_data = self.yield_size_on_capital(asset, dts, min(capital, control_capital))
        size = self.max_order_control.validate(asset, size, portfolio, dts)
        self.simulate_order(event, size, dts, direction)

    def simulate_order(self, event, amount, dts, direction):
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
        asset = event.asset
        size_array, order_data = self.calculate_size_arrays(asset, amount, dts)

        if asset.bid_mechanism:
            tickers = self.yield_tickers_on_size(len(size_array))
            ticker_prices = [order_data.minute[ticker] for ticker in tickers]
            iterator = zip(ticker_prices, tickers)
        else :
            # simulate_dist 已经包含剔除限制
            simulate_prices = self.simulate_dist(order_data, len(size_array))
            tickers = [locate_pos(price, order_data.minutes, direction) for price in simulate_prices]
            # ticker_price --- filter simulate_prices
            ticker_prices = np.clip(simulate_prices, order_data.minutes.min(), order_data.minutes.max())
            iterator = zip(ticker_prices, tickers)
        orders = [Order(event, amount, *args, self.execution_style,  self.slippage_model)
                  for amount, args in zip(size_array, iterator)]
        # simulate transactions
        final_orders = self._finalize(orders, order_data)
        return final_orders

    def yield_order(self, dts, event, price_array, size_array, ticker_array, direction, portfolio):
        # 买入订单（卖出 --- 买入）
        asset = event.asset
        size = self.max_order_control.validate(asset, sum(size_array), portfolio, dts)
        # 按比例进行scale
        control_size_array = map(lambda x: x * sum(size_array) / size, size_array)
        order_data = self._create_data(dts, asset)
        # 存在controls
        orders = [Order(event, *args, direction, self.execution_style, self.slippage_model)
                  for args in zip(price_array, control_size_array, ticker_array)]
        # simulate transactions
        final_orders = self._finalize(orders, order_data)
        return final_orders


if __name__ == '__main__':

    order_creator = OrderSimulation()
