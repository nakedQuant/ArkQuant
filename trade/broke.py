# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from itertools import chain
import numpy as np , pandas as pd
from collections import namedtuple
from multiprocessing import Pool
from functools import partial
from finance.transaction import create_transaction,simulate_transaction

OrderData = namedtuple('OrderData','minutes daily pct')


class OrderCreated(object):
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
                 multiplier = 2):
        self._data_protal = portal
        self._slippage = slippage
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

    def uncover_algo(self,asset,dts,capital):
        orderdata = self._create_BarData(dts,asset)
        #满足限制
        restricted_capital = orderdata.Preamount * self.fraction
        capital = capital if restricted_capital > capital else restricted_capital
        # 确保满足最低的一手
        per_capital = min([asset.tick_size * orderdata.preclose * asset.tick_size * (1+asset.restricted),self.base_capital])
        assert capital < per_capital , ValueError('capital must satisfy the base tick size')
        size = capital / per_capital
        self.per_capital = per_capital
        return size

    def _create_BarData(self,dt,asset):
        OrderData.minutes = self._data_protal.get_spot_value(dt,asset,'minute')
        OrderData.daily = self._data_protal.get_spot_value(dt,asset,'daily')
        OrderData.pct = self._data_protal.get_equity_pctchange(dt,asset)
        return OrderData

    def simulate_dist(self, data, size):
        """
        基于开盘的涨跌幅决定 --- 当天的概率分布
        simulate price distribution to place on transactions
        :param size: number of transactions
        :param raw:  data for compute
        :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        :return: array of simualtion price
        """
        pct = data.pct
        kline = data.daily
        preclose = kline['close'] / (1+pct)
        open_pct = kline['open'] / preclose
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if size > 0:
            #模拟价格分布
            dist = 1 + np.copysign(alpha,np.random.beta(alpha,100,size))
        else:
            dist = [1 + alpha / 100]
        clip_price = np.clip(dist,-0.1,0.1) * preclose
        return clip_price

    def _create_price_order(self,asset,size,data):
        """
            A股主板，中小板首日涨幅最大为44%而后10%波动，针对不存在价格笼子（科创板，创业板后期对照科创板改革）
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，
            不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的102%，卖出价格不得低于买入价格98%，
            申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        clip_price = self.simulate_dist(data,size)
        tiny_orders = [PriceOrder(asset,price,self.per_capital) for price in clip_price]
        return tiny_orders

    def _create_ticker_order(self,asset,size):
        """
            由于价格笼子，科创板可以参考基于时间的设置订单
        """
        order = []
        interval = 4 * 60 / size
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin'%interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin'%interval)
        #确保首尾
        tick_interval = list(chain(*zip(upper, bottom)))[:size - 1]
        tick_interval.append(pd.Timestamp('2020-06-17 14:57:00',freq='%dmin'%interval))
        for ticker in tick_interval:
            # 根据设立时间去定义订单
            ticker_order = TickerOrder(asset,ticker,self.per_capital)
            ticker_order = TickerOrder(asset,ticker,self.per_capital)
            order.append(ticker_order)
        return order

    def create_order(self,asset,sizes,dts):
        """
            基于amount 生成订单
        """
        if asset.bid_mechanism:
            orders = self._create_ticker_order(asset,len(sizes))
        else:
            data = self._create_BarData(asset, dts)
            orders = self._create_price_order(asset,len(sizes),data)
        return orders

    def simulate_transaction(self,asset,capital,dts,commission):
        """
            supplement , capital
        """
        size,orderdata = self.uncover_algo(asset,dts,capital)
        if asset.bid_mechanism :
            orders = self._create_ticker_order(asset,size,orderdata)
        else:
            orders = self._create_price_order(asset,size)

        txns = simulate_transaction(orders,orderdata,commission)
        return txns


class Internal(object):

    def __init__(self,
                 creator,
                 commission,
                 multiplier = 1 ,
                 delay = 1):
        self.creator = creator
        self.commission = commission
        self.multiplier = multiplier
        self.delay = delay

    def calculate_per_size(self,q,data):
        capital = self.commission.min_base_cost * self.multiplier
        per_size = min([int(capital / data.preclose),q])
        return per_size

    def intern_tunnel(self,p,c,dts):
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
        #计算卖出持仓
        asset = p.inner_position.asset
        #获取数据
        p_minutes = self.creator._create_BarData(dts,asset)
        # amount
        q = p.inner_position.amount
        # 每次卖出size
        p_size = self.calculate_per_size(q,p_minutes)
        #构建卖出组合
        size_array = np.tile([p_size],int(q/p_size))
        idx = np.random(int(q/p_size))
        size_array[idx] += q % p_size
        # 生成卖出订单
        p_orders = self.creator.create_order(asset, size_array)
        # 根据 p_orders ---- 生成对应成交的ticker
        p_transactions = [simulate_transaction(p_order,p_minutes,self.commission)
                          for p_order in p_orders]
        p_transaction_price = np.array([t.price for t in p_transactions])
        # 执行对应的买入算法
        #获取买入标的的数据 c
        c_minutes = self.creator._create_BarData(dts,c)
        # 增加ticker shift
        c_tickers = [pd.Timedelta(minutes='%dminutes'%self.delay) + t.ticker for t in p_transactions]
        c_ticker_price = np.array([c_minutes[ticker] for ticker in c_tickers])
        #计算买入数据基于价格比值
        times = p_transaction_price / c_ticker_price
        c_size = [np.floor(size * times) for size in size_array]
        #构建对应买入订单
        c_transactions = [create_transaction(c,c_size,c_price,c_ticker)
                          for c_price, c_ticker in
                          zip(c_ticker_price,c_tickers)]
        return p_transactions,c_transactions


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
                 internal,

                 allocation):
        self.internal = internal
        self.allocation = allocation

    def _eval_consecutive_procesure(self,objs,dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        :return:
        """
        tunnel  = self.internal.intern_tunnel
        p_func = partial(tunnel,dts = dts)

        with Pool(processes=len(objs))as pool:
            results = [pool.apply_async(p_func,*obj)
                       for obj in objs]
            txns = chain(*results)
        return txns

    def _eval_simple_procesure(self,supplements,capital,dts):
        capital_mappings = self.allocation.compute(supplements.values(),capital)
        p_func = partial(self.internal.creator.simulate_transaction(
                                                                    dts = dts,
                                                                    commission = self.commission))
        with Pool(processes= len(supplements)) as pool:
            result = [pool.apply_async(p_func,asset,capital_mappings[asset])
                      for asset in supplements.values()]
        transactions = chain(*result)
        return transactions

    def carry_out(self, engine, ledger):
        """建立执行计划"""
        objs, supplements, capital, dts = engine.execute_engine(ledger)
        dual_txns = self._eval_consecutive_procesure(objs)
        txns = self._eval_simple_procesure(supplements,capital,dts)
        return dual_txns,txns

    def evaluate_efficiency(self,capital,puts,dts):
        """
            根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        """