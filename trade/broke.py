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
from abc import ABC,abstractmethod
from finance.transaction import create_transaction
from finance.order import  Order,PriceOrder
from utils.dt_utilty import  locate_pos

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
        OrderData.bar = self._data_protal.get_spot_value(dt,asset,'minute')
        OrderData.kline = self._data_protal.get_spot_value(dt,asset,'daily')
        OrderData.pre = self._data_protal.get_prevalue(dt,asset,'daily')
        OrderData.pct = self._data_protal.get_equity_pct(dt,asset)
        return OrderData

    @staticmethod
    def yield_tickers_on_size(size):
        interval = 4 * 60 / size
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        # 确保首尾
        tick_interval = list(chain(*zip(upper, bottom)))[:size - 1]
        tick_interval.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_interval

    def yield_size_on_capital(self,asset,dts,capital):
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
        kline = data.kline
        preclose = kline['close'] / (1 + data.pct)
        open_pct = kline['open'] / preclose
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

    def simulate_order(self,asset,capital,dts):
        """
            supplement , capital
        """
        size,orderdata = self.yield_tickers_on_size(asset,dts,capital)
        if asset.bid_mechanism :
            trigger_orders = self._create_ticker_order(asset,size,orderdata)
        else:
            restricted = asset.restricted(dts)
            trigger_orders = self._create_price_order(asset,size,orderdata,restricted)
        return trigger_orders

    def simulate(self,asset,sizes,dts,direction):
        """
            基于amount 生成订单
        """
        data = self._create_data(asset, dts)
        if asset.bid_mechanism :
            tickers = self.yield_tickers_on_size(len(sizes))
            ticker_prices = [data.min[ticker] for ticker in tickers]
            iterator = zip(ticker_prices,tickers)
        else :
            #simulate_dist 已经包含剔除限制
            simulate_prices = self.simulate_dist(data,len(sizes))
            tickers = [locate_pos(price,data.minutes,direction) for price in simulate_prices]
            iterator = zip(simulate_prices,tickers)
        orders = [Order(asset,amount,*args,self._slippage) for amount,args in zip(sizes,iterator)]
        trigger_orders = [order for order in orders if order.check_trigger(data.pre['close'])]
        return trigger_orders


class Internal(object):

    def __init__(self,
                 creator,
                 commission,
                 multiplier = 1,
                 delay = 1):
        self.creator = creator
        self.commission = commission
        self.multiplier = multiplier
        self.delay = delay

    def calculate_per_size(self,q,data):
        capital = self.commission.min_base_cost * self.multiplier
        per_size = min([int(capital / data.pre['close']),q])
        return per_size

    @staticmethod
    def create_bulk_transactions(orders,fee):
        txns = [create_transaction(order,fee) for order in orders]
        return txns

    def interactive(self,asset,capital,dts):
        """
            基于capital --- 买入标的
        """
        orders = self.creator.simulate_order(asset,capital,dts)
        #将orders --- transactions
        fee = self.commission.calculate_rate(asset,'positive',dts)
        txns = self.create_bulk_transactions(orders,fee)
        #计算效率
        efficiency = sum([order.per_capital for order in orders]) / capital
        return (txns,efficiency)

    def interactive_tunnel(self,p,c,dts):
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
        selector = self.creator._create_data
        #卖出持仓
        asset = p.inner_position.asset
        q = p.inner_position.amount
        #获取数据并计算每次卖出size
        p_data = selector(dts,asset)
        #根据size拆分order
        p_size = self.calculate_per_size(q,p_data)
        size_array = np.tile([p_size],int(q/p_size))
        size_array[ np.random(int(q/p_size))] += q % p_size
        # 构建对应卖出订单并生成对应txn
        p_orders = self.creator.simulate(asset, size_array,dts,'negative')
        p_fee = self.commission.calculate_rate(asset,'negative',dts)
        p_transactions = self.create_bulk_transactions(p_orders,p_fee)
        #计算效率
        p_uility = sum([order.amount for order in p_orders]) / q

        # 执行对应的买入算法
        # c_data = self.creator._create_data(dts,c)
        c_data = selector(dts,c)
        # 切换之间存在时间差，默认以minutes为单位
        c_tickers = [pd.Timedelta(minutes='%dminutes'%self.delay) + t._created_dt for t in p_transactions]
        #根据ticker价格比值 --- 拆分订单
        c_ticker_prices = np.array([c_data[ticker] for ticker in c_tickers])
        p_transaction_prices = np.array([t.price for t in p_transactions])
        times = p_transaction_prices / c_ticker_prices
        c_sizes = [np.floor(size * times) for size in size_array]
        #构建对应买入订单并生成对应txn
        c_orders = self.creator.simulate(c, c_sizes,dts,'positive')
        c_fee = self.commission_rate(c,'positive',dts)
        c_transactions = self.create_bulk_transactions(c_orders,c_fee)
        #计算效率
        c_uility = sum([order.amount for order in c_orders]) / sum(c_sizes)
        return (p_transactions,p_uility,c_transactions,c_uility)


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

    def _eval_consecutive_procesure(self,items,dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        :return:
        """
        tunnel  = self.internal.intern_tunnel
        p_func = partial(tunnel,dts = dts)
        with Pool(processes=len(items))as pool:
            results = [pool.apply_async(p_func,*obj)
                       for obj in items]
            #卖出订单，买入订单分开来
            p_txns,p_uility,c_txns,c_uility = zip(*results)
        return p_txns,p_uility,c_txns,c_uility

    def _eval_procesure(self,supplements,capital,dts):
        capital_mappings = self.allocation.compute(supplements.values(),capital)
        p_func = partial(self.internal.interactive(dts = dts))
        with Pool(processes= len(supplements)) as pool:
            result = [pool.apply_async(p_func,asset,capital_mappings[asset])
                      for asset in supplements.values()]
        # transaction , efficiency
        transactions,uility = list(zip(*result))
        return transactions , uility

    def carry_out(self, simple_engine,ledger):
        """建立执行计划"""
        objs, supplements, capital, dts = simple_engine.execute_engine(ledger)
        p_txns,p_uility,c_txns,c_uility = self._eval_consecutive_procesure(objs)
        txns , uility = self._eval_procesure(supplements,capital,dts)
        # 根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        uility_ratio = np.mean(p_uility + c_uility + uility)
        transactions = p_txns + c_txns + txns
        return transactions , uility_ratio