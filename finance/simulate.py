#撮合模块
import math,uuid,numpy as np,pandas as pd, datetime as dt
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import OrderedDict
from numpy import isfinite
from enum import Enum
from toolz import valmap
from itertools import chain


class MatchUp(object):
    """ 撮合成交
        如果open_pct 达到10% --- 是否买入
        分为不同的模块 创业板，科创板，ETF
        包含 --- sell orders buy orders 同时存在，但是buy_orders --- 基于sell orders 和 ledger
        通过限制买入capital的pct实现分布买入
        但是卖出订单 --- 通过追加未成交订单来实现
        如何连接卖出与买入模块

        由capital --- calculate orders 应该属于在统一模块 ，最后将订单 --- 引擎生成交易 --- 执行计划式的，非手动操作类型的
        剔除ReachCancel --- 10%
        剔除SwatCancel --- 黑天鹅
    """
    def __init__(self,
                multiplier = 5,
                cancel_policy = [ReachCancel,SwatCancel],
                execution_style = MarketOrder,
                slippageModel = MarketImpact):

        #确定订单类型默认为市价单
        self.style = execution_style
        self.commission = AssetCommission(multiplier)
        self.cancel_policy = ComposedCancel(cancel_policy)
        self.engine = Engine()
        self.adjust_array = AdjustArray()
        self.record_transactions = OrderedDict()
        self.record_efficiency = OrderedDict()
        self.prune_closed_assets = OrderedDict()

    @property
    def _fraction(self):
        """设立成交量限制，默认为前一个交易日的百分之一"""
        return 0.05

    @_fraction.setter
    def _fraction(self,val):
        return val

    def load_data(self,dt,assets):
        raw = self.adjust_array.load_array_for_sids(dt,0,['open','close','volume','amount','pct'],assets)
        volume = { k : v['volume'] for k,v in raw.items()}
        if raw:
            """说明历史回测 --- 存在数据"""
            preclose = { k: v['close'] / (v['pct'] +1 ) for k,v in raw.items()}
            open_pct = { k: v['open'] / preclose[k] for k,v in raw.items()}
        else:
            """实时回测 , 9.25之后"""
            raw = self.adjust_array.load_pricing_adjusted_array(dt,2,['open','close','pct'],assets)
            minutes = self.adjust_array.load_minutes_for_sid(assets)
            if not minutes:
                raise ValueError('时间问题或者网路问题')
            preclose = { k : v['close'][-1]  for k,v in raw.items() }
            open_pct = { k : v.iloc[0,0] / preclose[k] for k,v in minutes.items()}
        dct = {'preclose':preclose,'open_pct':open_pct,'volume':volume}
        return dct

    #获取日数据，封装为一个API(fetch process flush other api)
    # def _create_bar_data(self, universe_func):
    #     return BarData(
    #         data_portal=self.data_portal,
    #         simulation_dt_func=self.get_simulation_dt,
    #         data_frequency=self.sim_params.data_frequency,
    #         trading_calendar=self.algo.trading_calendar,
    #         restrictions=self.restrictions,
    #         universe_func=universe_func
    #     )

    # def _load_tick_data(self,asset,ticker):
    #     """获取当天实时的ticer实点的数据，并且增加一些滑加，+/-0.01"""
    #     minutes = self.adjust_array.load_minutes_for_sid(asset,ticker)
    #     return minutes

    def execute_cancel_policy(self,target):
        """买入 --- 如果以涨停价格开盘过滤，主要针对买入标的"""
        _target =[self.cancel_policy.should_cancel(item) for item in target]
        result = _target[0] if _target else None
        return result

    def _restrict_buy_rule(self,dct):
        """
            主要针对于买入标的的
            对于卖出的，遵循最大程度卖出
        """
        self.capital_limit = valmap(lambda x : x * self._fraction,dct)

    def attach_pruned_holdings(self,puts,holdings):
        closed_holdings = valfilter(lambda x: x.inner_position.asset in self.prune_closed_assets, holdings)
        puts.update(closed_holdings)
        return puts

    def carry_out(self,engine,ledger):
        """建立执行计划"""
        #engine --- 获得可以交易的标的
        puts, calls,holdings,capital,dts = engine.execute_engine(ledger)
        #将未完成的卖出的标的放入puts
        puts = self.attach_pruned_holdings(puts,holdings)
        self.commission._init_base_cost(dts)
        #获取计算订单所需数据
        assets = set([position.inner_position.asset for position in holdings]) | set(chain(*calls.values()))
        raw = self.load_data(dts,assets)
        #过滤针对标的
        calls = valmap(lambda x:self.execute_cancel_policy(x),calls)
        calls = valfilter(lambda x : x is not None,calls)
        call_assets = list(calls.values())
        #已有持仓标的
        holding_assets = [holding.inner_position.asset for holding in holdings]
        #卖出持仓标的
        put_assets = [ put.inner_position.asset for put in puts]
        # 限制 --- buys_amount,sell --- volume
        self._restrict_rule(raw['amount'])
        #固化参数
        match_impl = partial(self.positive_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
        _match_impl = partial(self.dual_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
        _match_impl(put_assets,call_assets) if puts else match_impl(call_assets)
        #获取存量的transactions
        final_txns = self._init_engine(dts)
        #计算关于总的订单拆分引擎撮合成交的的效率
        self.evaluate_efficiency(capital,puts,dts)
        #将未完成需要卖出的标的继续卖出
        self.to_be_pruned(puts)
        return final_txns

    def _init_engine(self,dts):
        txns = self.engine.engine_transactions
        self.record_transactions[dts] = txns
        self.engine.reset()
        return txns

    def evaluate_efficiency(self,capital,puts,dts):
        """
            根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        """
        txns = self.record_transactions[dts]
        call_efficiency = sum([ txn.amount * txn.price for txn in txns if txn.amount > 0 ]) / capital
        put_efficiency = sum([txn.amount for txn in txns if txn.amount < 0]) / \
                         sum([position.inner_position.amount for position in puts.values()]) if puts else 0
        self.record_efficiency[dts] = {'call':call_efficiency,'put':put_efficiency}

    def to_be_pruned(self,dts,puts):
        #将未完全卖出的position存储继续卖出
        txns = self.record_transactions[dts]
        txn_put_amount = {txn.asset:txn.amount for txn in txns if txn.amount < 0}
        position_amount = {position.inner_position.asset : position.inner_position.amount for position in puts}
        pct = txn_put_amount / position_amount
        uncompleted = keyfilter(lambda x : x < 1,pct)
        self.prune_closed_assets[dts] = uncompleted.keys()

    def positive_match(self,calls,holdings,capital,raw,dts):
        """buys or sells parallel"""
        if calls:
            capital_dct = self.policy.calculate(calls,capital,dts)
        else:
            capital_dct = self.policy.calculate(holdings, capital,dts)
        #买入金额限制
        restrict_capital = {asset : self.capital_limit[asset] if capital >= self.capital_limit[asset]
                                    else capital  for asset ,capital in capital_dct.items()}

        call_impl = partial(self.engine.call,raw = raw,min_base_cost = self.commission.min_base_cost)
        with Pool(processes=len(restrict_capital))as pool:
            results = [pool.apply_async(call_impl,asset,capital)
                       for asset,capital in restrict_capital.items()]
            txns = chain(*results)
        return txns

    def dual_match(self,puts,calls,holdings,capital,dts,raw):
        #双向匹配
        """基于capital生成priceOrder"""
        txns = dict()
        if calls:
            capital_dct = self.policy.calculate(calls,capital,dts)
        else:
            left_holdings = set(holdings) - set(puts)
            capital_dct = self.policy.calculate(left_holdings,capital,dts)
        #call orders
        txns['call'] = self.positive_match(calls,holdings,capital_dct,raw,dts)
        #put orders
        # --- 直接以open_price卖出;如果卖出的话 --- 将未成交的卖出订单orders持续化
        for txn_capital in self.engine.put(puts,calls,raw,self.commission.min_base_cost):
            agg = sum(capital_dct.values())
            trading_capital = valmap(lambda x : x * txn_capital / agg,capital_dct )
            self.engine._infer_order(trading_capital)

