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
        #当MatchUp运行结束之后
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
