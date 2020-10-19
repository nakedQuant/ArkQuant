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
from gateway.driver.data_portal import portal
from finance.control import UnionControl
from finance.order import Order, PriceOrder, TickerOrder, transfer_to_order
from finance.transaction import create_transaction
from util.dt_utilty import locate_pos


class Simulation(ABC):

    @staticmethod
    @abstractmethod
    def simulate_dist(num, open_pct):
        raise NotImplementedError('distribution of price')

    @staticmethod
    @abstractmethod
    def simulate_ticker(self, num):
        raise NotImplementedError('simulate ticker of trading_day')


class SimpleSimulation(Simulation):

    @staticmethod
    def simulate_dist(num, open_pct):
        """模拟价格分布，以开盘振幅为参数"""
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        if num > 0:
            # 模拟价格分布
            dist = 1 + np.copysign(alpha, np.random.beta(alpha, 100, num))
        else:
            dist = [1 + alpha / 100]
        return dist

    @staticmethod
    def simulate_ticker(num):
        # ticker arranged on sequence
        interval = 4 * 60 / num
        # 按照固定时间去执行
        upper = pd.date_range(start='09:30', end='11:30', freq='%dmin' % interval)
        bottom = pd.date_range(start='13:00', end='14:57', freq='%dmin' % interval)
        # 确保首尾
        tick_intervals = list(chain(*zip(upper, bottom)))[:num - 1]
        tick_intervals.append(pd.Timestamp('2020-06-17 14:57:00', freq='%dmin' % interval))
        return tick_intervals


class BaseDivision(ABC):

    @lru_cache(maxsize=32)
    def _init_data(self, asset, dts):
        open_pct, pre_close = portal.get_open_pct(asset, dts)
        return open_pct, pre_close

    @abstractmethod
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
    """
    def __init__(self,
                 slippage,
                 execution_style,
                 distribution=SimpleSimulation()):
        self.slippage_model = slippage
        self.execution_style = execution_style
        self.dis = distribution

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
        size_array[np.random.randint(0, len(size_array),size % per_size)] += 1
        size_array = size_array * asset.tick_size
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
        size_array = size_array * asset.tick_size
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
    """

    def __init__(self,
                 slippage,
                 execution_style,
                 distribution=SimpleSimulation):
        self.slippage_model = slippage
        self.execution_style = execution_style
        self.dis = distribution

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
        size_array = size_array * -1
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


class BlotterSimulation(object):
    """
        transform orders which are simulated by gen module to transactions
        撮合成交逻辑基于时间或者价格
    """
    def __init__(self,
                 commission_model,
                 slippage_model,
                 execution_model):
        self.commission = commission_model
        self.slippage = slippage_model
        self.execution = execution_model

    def _trigger_check(self, order, dts):
        """
            trigger orders checked by execution_style and slippage_style
        """
        assert isinstance(order, Order), 'unsolved order type'
        asset = order.asset
        price = order.price
        # 基于订单的设置的上下线过滤订单
        upper = 1 + self.execution.get_limit_ratio(asset, dts)
        bottom = 1 - self.execution.get_stop_ratio(asset, dts)
        if bottom <= price <= upper:
            # 计算滑价系数
            slippage = self.slippage.calculate_slippage_factor(asset, dts)
            order.price = price * (1+slippage)
            return order
        return False

    def _validate(self, order, dts):
        # fulfill the missing attr of PriceOrder and TickerOrder
        asset = order.asset
        price = order.price
        direction = np.sign(order.amount)
        minutes = portal.get_spot_value(dts, asset, 'minutes', ['close'])
        if isinstance(order, PriceOrder):
            ticker = locate_pos(price, minutes, direction)
            new_order = transfer_to_order(order, ticker=ticker)
        elif isinstance(order, TickerOrder):
            ticker = order.created_dt
            price = minutes['close'][ticker] if isinstance(ticker, int) \
                else minutes['close'][int(ticker.timestamp())]
            new_order = transfer_to_order(order, price=price)
        elif isinstance(order, Order):
            new_order = order
        else:
            raise ValueError
        new_order = self._trigger_check(new_order, dts)
        return new_order

    def create_transaction(self, orders, dts):
        trigger_orders = [order for order in orders if self._validate(order, dts)]
        # create txn
        transactions = [create_transaction(order_obj, self.commission) for order_obj in trigger_orders]
        return transactions


class Interactive(object):
    """
        transfer short to long on specific pipeline
    """
    def __init__(self, delay):
        self.delay = delay
        self.holidng_division = PositionDivision()
        self.capital_division = CapitalDivision()
        self.blotter = BlotterSimulation()

    def yield_interactive(self, position, asset, dts):
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
        short_transactions = self.yield_position(position, dts)
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
        long_transactions = self.blotter.create_transaction(orders, dts)
        return short_transactions, long_transactions

    def yield_capital(self, asset, capital, dts):
        capital_orders = self.capital_division.simulate_iterator(asset, capital, dts)
        capital_transactions = self.blotter.create_transaction(capital_orders, dts)
        return capital_transactions

    def yield_position(self, position, dts):
        holding_orders = self.holidng_division.simulate_iterator(position, dts)
        holding_transactions = self.blotter.create_transaction(holding_orders, dts)
        return holding_transactions


class Generator(object):
    """
        a. calculate amount to determin size
        b. create ticker_array depend on size
        c. simulate order according to ticker_price , ticker_size , ticker_price
            --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
            --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
        d. principle:
            a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
            b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
            c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
        e.
            combine simpleEngine and blotter module
            engine --- asset --- orders --- transactions
            订单 --- 引擎生成交易 --- 执行计划式的

    """
    def __init__(self,
                 controls,
                 risk_model,
                 delay=1):
        self.interactive = Interactive(delay=delay)
        self.risk_model = risk_model
        self.trade_controls = UnionControl(controls)

    def engaged_in_risk(self, assets, capital, portfolio, dts):
        allocation = self.risk_model.compute(assets, capital, dts)
        for k, v in allocation.items():
            asset, amount, capital = self.trade_controls.validate(k, 0, v, portfolio, dts)

    def enroll_implement(self, asset, capital, dts):
        """基于资金买入对应仓位"""
        self.interactive.yield_capital(asset, capital, dts)

    def withdraw_implement(self, positions, dts):
        """单独的卖出仓位"""
        self.interactive.yield_position(positions, dts)

    def dual_implement(self, duals, dts):
        """
            针对一个pipeline算法，卖出 -- 买入
        """
        p, asset = duals
        self.interactive.yield_interactive(p, asset, dts)

    def carry_out(self, engine, ledger):
        """建立执行计划"""
        dts, capital, negatives, dual, positives = engine.execute_algorithm(ledger)
        # 直接买入
        # 直接卖出
        # 卖出 --- 买入
        # unchanged_positions
        # 根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        # --- 通过portfolio的资金使用效率来度量算法的效率

        # 执行成交
        # ledger.process_transaction(transactions)
