# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
import numpy as np , pandas as pd
from collections import namedtuple
from abc import ABC,abstractmethod
from finance.oms.order import Order,PriceOrder
from utils.dt_utilty import locate_pos

OrderData = namedtuple('OrderData','min kline pre pct')


class BaseCreated(ABC):

    @abstractmethod
    def yield_tickers_on_size(self,size):
        """
            根据size在交易时间段构建ticker组合
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def yield_size_on_capital(self,asset,dts,capital):
        """
            基于资金限制以及preclose计算隐藏的订单个数
        """
        raise NotImplementedError()

    @abstractmethod
    def simulate_dist(self,*args):
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
                 execution_style,
                 multiplier = 2):
        self._data_protal = portal
        self._slippage = slippage
        self._style = execution_style
        self.multiplier = multiplier

    @property
    def min_base_cost(self):
        """为保证满足最小交易成本 --- e.g : 5所需的capital """
        return self.min_base_cost

    @min_base_cost.setter
    def fraction(self,val):
        return val

    @property
    def base_capital(self):
        return  self.multiplier * self.min_base_cost

    @property
    def fraction(self):
        """设立成交量限制，默认为前一个交易日的百分之一"""
        return 0.05

    @fraction.setter
    def fraction(self,val):
        return val

    def _create_data(self,dt,asset):
        """生成OrderData"""
        OrderData.bar = self._data_protal.get_spot_value(dt,asset,'minute')
        OrderData.close = self._data_protal.get_spot_value(dt,asset,'daily')
        OrderData.open_pct = self._data_protal.get_open_pct(dt,asset)
        return OrderData

    def calculate_size_arrays(self,asset,dts,q):
        """根据目标q --- 生成size序列拆分订单数量"""
        data = self._create_data(dts, asset)
        capital = self.commission.min_base_cost * self.multiplier
        per_size = min([int(capital / data.pre['close']),q])
        size_array = np.tile([per_size],int(q/per_size))
        size_array[ np.random(int(q/per_size))] += q % per_size
        return per_size , data

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

    def yield_size_on_capital(self,asset,dts,capital):
        """根据capital 生成资金订单"""
        orderdata = self._create_data(dts,asset)
        #满足限制
        restricted_capital = orderdata.pre['volume'] * self.fraction
        capital = capital if restricted_capital > capital else restricted_capital
        # 确保满足最低的一手
        per_capital = min([asset.tick_size * orderdata.pre['close'] * asset.tick_size * (1+asset.restricted),self.base_capital])
        assert capital < per_capital , ValueError('capital must satisfy the base tick size')
        size = capital / per_capital
        self.per_capital = per_capital
        return size , orderdata

    def simulate_dist(self,data, size, restricted):
        """模拟价格分布，以开盘振幅为参数"""
        open_pct = data.open_pct
        preclose = data.close / open_pct
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if size > 0:
            #模拟价格分布
            dist = 1 + np.copysign(alpha,np.random.beta(alpha,100,size))
        else:
            dist = [1 + alpha / 100]
        #避免跌停或者涨停
        clip_pct = np.clip(dist,(1 - restricted ),(1+restricted))
        sim_prices = clip_pct * preclose
        return sim_prices

    def _create_price_order(self,asset,size,data,restricted):
        """
            A股主板，中小板首日涨幅最大为44%而后10%波动，针对不存在价格笼子（科创板，创业板后期对照科创板改革）
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，
            不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的102%，卖出价格不得低于买入价格98%，
            申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        clip_prices = self.simulate_dist(data,size,restricted)
        tickers = [locate_pos(price,data.minutes,'positive') for price in clip_prices]
        orders = [PriceOrder(asset,self.per_capital,price,ticker,self._style,self._slippage)
                  for price,ticker in zip(clip_prices,tickers)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre['close'])]
        return trigger_orders

    def _create_ticker_order(self,asset,size,data):
        """
            由于价格笼子，科创板可以参考基于时间的设置订单
        """
        ticker_intervals = self.yield_tickers_on_size(size)
        ticker_prices = [data.min[ticker] for ticker in ticker_intervals]
        orders = [PriceOrder(asset,self.per_capital,price,ticker,self._style,self._slippage)
                  for price , ticker in zip(ticker_prices,ticker_prices)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre['close'])]
        return trigger_orders

    def simulate_capital_order(self,asset,capital,dts):
        """
            capital --- order
        """
        size,orderdata = self.yield_tickers_on_size(asset,dts,capital)
        if asset.bid_mechanism :
            trigger_orders = self._create_ticker_order(asset,size,orderdata)
        else:
            restricted = asset.restricted(dts)
            trigger_orders = self._create_price_order(asset,size,orderdata,restricted)
        return trigger_orders

    def simulate_order(self,asset,size_array,data,direction):
        """
            基于amount 生成订单
        """
        if asset.bid_mechanism :
            tickers = self.yield_tickers_on_size(len(size_array))
            ticker_prices = [data.min[ticker] for ticker in tickers]
            iterator = zip(ticker_prices,tickers)
        else :
            #simulate_dist 已经包含剔除限制
            simulate_prices = self.simulate_dist(data,len(size_array))
            tickers = [locate_pos(price,data.minutes,direction) for price in simulate_prices]
            iterator = zip(simulate_prices,tickers)
        orders = [Order(asset,amount,*args,self._slippage) for amount,args in zip(size_array,iterator)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre['close'])]
        return trigger_orders
