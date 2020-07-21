# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC , abstractmethod
from toolz import valmap
from itertools import chain


class Engine(ABC):
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
    def reset(self):
        self.engine_transactions = []

    @abstractmethod
    def _create_orders(self,asset,raw,**kwargs):
        """
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
            102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        raise NotImplementedError

    @abstractmethod
    def simulate_dist(self,alpha,size):
        """
        simulate price distribution to place on transactions
        :param size: number of transactions
        :param raw:  data for compute
        :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        :return: array of simualtion price
        """
        raise NotImplementedError

    def market_orders(self, capital, asset):
        """按照市价竞价按照size --- 时间分割 TickerOrder"""
        min_base_cost = self.commission.min_base_cost
        size = capital / min_base_cost
        tick_interval = self.simulate_dist(size)
        for dts in tick_interval:
            # 根据设立时间去定义订单
            order = TickerOrder(asset,dts,min_base_cost)
            self.oms(order, eager=True)

    def call(self, capital, asset,raw,min_base_cost):
        """执行前固化的订单买入计划"""
        if not asset.bid_rule:
            """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单 PriceOrder"""
            under_orders = self._create_orders(asset,
                                               raw,
                                               capital=capital,
                                               min_base_cost = min_base_cost)
        else:
            under_orders = self.market_orders(capital, asset)
        #执行买入订单
        self.internal_oms(under_orders)

    def _infer_order(self,capital_dct):
        """基于时点执行买入订单,时间为进行入OMS系统的时间 --- 为了衔接卖出与买入"""
        orders = [RealtimeOrder(asset,capital) for asset,capital in capital_dct]
        self.oms(orders)

    def _put_impl(self,position,raw,min_base_cost):
        """按照市价竞价"""
        amount = position.inner_position.amount
        asset = position.inner_position.asset
        last_sync_price = position.inner_position.last_sync_price
        if not asset.bid_rule:
            """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单"""
            tiny_put_orders = self._create_orders(asset,
                                                  raw,
                                                  amount = amount,
                                                  min_base_cost = min_base_cost)
        else:
            min_base_cost = self.commission.min_base_cost
            per_amount = np.ceil(self.multiplier['put'] * min_base_cost / (last_sync_price * 100))
            size = amount / per_amount
            #按照size --- 时间分割
            intervals = self.simulate_tick(size)
            for dts in intervals:
                tiny_put_orders = TickerOrder(per_amount * 100,asset,dts)
        return tiny_put_orders

    @staticmethod
    def simulate_tick(size,final = True):
        interval = 4 * 60 / size
        # 按照固定时间去执行
        day_m = pd.date_range(start='09:30', end='11:30', freq='%dmin'%interval)
        day_a = pd.date_range(start='13:00', end='14:57', freq='%dmin'%interval)
        day_ticker = list(chain(*zip(day_m, day_a)))
        if final:
            last = pd.Timestamp('2020-06-17 14:57:00',freq='%dmin'%interval)
            day_ticker.append(last)
        return day_ticker

    def put(self,puts,raw,min_base_cost):
        put_impl = partial(self._put_impl,
                           raw = raw,
                           min_base_cost = min_base_cost)
        with Pool(processes=len(puts))as pool:
            results = [pool.apply_async(put_impl,position)
                       for position in puts.values]
            put_orders = chain(*results)
            # 执行卖出订单 --- 返回标识
        for txn in self.internal_oms(put_orders, dual=True):
                #一旦有订单成交 基于队虽然有延迟，但是不影响
                txn_capital = txn.amount * txn.price
                yield txn_capital

    @abstractmethod
    def internal_oms(self,orders,eager = True):
        """
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
        """
        raise NotImplementedError


class BackEngine(Engine):
    """
        基于ticker --- 进行回测,在执行具体的买入标的基于ticker数据真实模拟
    """
    def __init__(self,
                multiplier = {'call':1.5,'put':2},
                slippageModel = MarketImpact):

        # multipiler --- 针对基于保持最低交易成的capital的倍数进行交易
        self.multiplier = multiplier
        self.slippage = slippageModel()
        self.engine_transactions = []

    def _create_orders(self,asset,raw,**kwargs):
        """
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
            102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        multiplier = self.multiplier['call']
        min_base_cost = kwargs['min_base_cost']
        preclose = raw['preclose'][asset]
        open_pct = raw['open_pct'][asset]
        volume_restriction = self.volume_limit[asset]
        try:
            capital = kwargs['capital']
            #ensuer_amount --- 手
            bottom_amount = np.floor(capital / (preclose * 110))
            if bottom_amount == 0:
                raise ValueError('satisfied at least 100 stocks')
            #是否超过限制
            ensure_amount = bottom_amount if bottom_amount <= volume_restriction else volume_restriction
        except KeyError:
            amount = kwargs['amount']
            ensure_amount = amount if amount <= volume_restriction else volume_restriction
        # 计算拆分订单的个数，以及单个订单金额
        min_per_value = 90 * preclose / (open_pct + 1)
        ensure_per_amount = np.ceil(multiplier * min_base_cost / min_per_value)
        # 模拟价格分布的参数 --- 个数 数据 滑价系数
        size = ensure_amount // ensure_per_amount
        # volume = raw['volume'][asset]
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        sim_pct = self.simulate_dist(abs(alpha),size)
        # 限价原则 --- 确定交易执行价格 针对于非科创板，创业板股票
        # limit = self.style.get_limit_price() if self.style.get_limit_price() else asset.price_limit(dts)
        # stop = self.style.get_stop_price() if self.style.get_stop_price() else asset.price_limit(dts)
        limit = self.style.get_limit_price() if self.style.get_limit_price() else 0.1
        stop = self.style.get_stop_price() if self.style.get_stop_price() else 0.1
        clip_price = np.clip(sim_pct,-stop,limit) * preclose
        # 将多余的手分散
        sim_amount = np.tile([ensure_per_amount], size) if size > 0 else [ensure_amount]
        random_idx = np.random.randint(0, size, ensure_amount % ensure_per_amount)
        for idx in random_idx:
            sim_amount[idx] += 1
        #形成订单
        tiny_orders =  [PriceOrder(asset,args[0],args[1])
                     for args in zip(sim_amount,clip_price)]
        return tiny_orders

    def simulate_dist(self,alpha,size):
        """
        simulate price distribution to place on transactions
        :param size: number of transactions
        :param raw:  data for compute
        :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        :return: array of simualtion price
        """
        # 涉及slippage --- 基于ensure_amount --- multiplier
        if size > 0:
            #模拟价格分布
            dist = 1 + np.copysign(alpha,np.random.beta(alpha,100,size))
        else:
            dist = [1 + alpha  / 100]
        return dist

    def internal_oms(self,orders,eager = True):
        """
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
        """
        raise NotImplementedError()

    def _cleanup_expired_assets(self, dt, position_assets):
        """
        Clear out any assets that have expired before starting a new sim day.

        Performs two functions:

        1. Finds all assets for which we have open orders and clears any
           orders whose assets are on or after their auto_close_date.

        2. Finds all assets for which we have positions and generates
           close_position events for any assets that have reached their
           auto_close_date.
        """
        algo = self.algo

        def past_auto_close_date(asset):
            acd = asset.auto_close_date
            return acd is not None and acd <= dt

        # Remove positions in any sids that have reached their auto_close date.
        assets_to_clear = \
            [asset for asset in position_assets if past_auto_close_date(asset)]
        metrics_tracker = algo.metrics_tracker
        data_portal = self.data_portal
        for asset in assets_to_clear:
            metrics_tracker.process_close_position(asset, dt, data_portal)

        # Remove open orders for any sids that have reached their auto close
        # date. These orders get processed immediately because otherwise they
        # would not be processed until the first bar of the next day.
        blotter = algo.blotter
        assets_to_cancel = [
            asset for asset in blotter.open_orders
            if past_auto_close_date(asset)
        ]
        for asset in assets_to_cancel:
            blotter.cancel_all_orders_for_asset(asset)

        # Make a copy here so that we are not modifying the list that is being
        # iterated over.
        for order in copy(blotter.new_orders):
            if order.status == ORDER_STATUS.CANCELLED:
                metrics_tracker.process_order(order)
                blotter.new_orders.remove(order)


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
                cancel_policy,
                execution_style,
                slippageMode):

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
