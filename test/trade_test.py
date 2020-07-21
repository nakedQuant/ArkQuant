# class ExecutionStyle(ABC):
#     """
#         base class for order execution style
#     """
#     @abstractmethod
#     def get_limit_price(self, is_buy):
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_stop_price(self, is_buy):
#         raise NotImplementedError
#
#
# class MarketOrder(ExecutionStyle):
#
#     def __init__(self, exchange=None):
#         self._exchange = exchange
#
#     def get_limit_price(self, _is_buy):
#         return None
#
#     def get_stop_price(self, _is_buy):
#         return None
#
#
# class LimitOrder(ExecutionStyle):
#     """
#         limit price --- maximum price for buys or minimum price for sells
#     """
#
#     def __init__(self, limit_price, asset=None, exchange=None):
#         check_stoplimit_prices(limit_price, 'limit')
#
#         self.limit_price = limit_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return asymmetric_round_price(self.limit_price, is_buy,
#                                       tick_size=(0.01 if self.asset is None else self.asset.tick_size))
#
#     def get_stop_price(self, _is_buy):
#         return None
#
#
# class StopOrder(ExecutionStyle):
#     """
#         stop_price ---- for sells the order will be placed if market price falls below this value .
#         for buys ,the order will be placed if market price rise above this value.
#     """
#
#     def __init__(self, stop_price, asset=None, exchange=None):
#         check_stoplimit_prices(stop_price, 'stop')
#
#         self.stop_price = stop_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return None
#
#     def get_stop_price(self, is_buy):
#         return asymmetric_round_price(
#             self.stop_price,
#             not is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
# class StopLimitOrder(ExecutionStyle):
#     """
#         price reach a threahold
#     """
#
#     def __init__(self, limit_price, stop_price, asset=None, exchange=None):
#         check_stoplimit_prices(limit_price, 'limit')
#         check_stoplimit_prices(stop_price, 'stop')
#
#         self.limit_price = limit_price
#         self.stop_price = stop_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return asymmetric_round_price(
#             self.limit_price,
#             is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
#     def get_stop_price(self, is_buy):
#         return asymmetric_round_price(
#             self.stop_price,
#             not is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
#
#
# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#         for limit_price ,this means preferring to round down on buys and preferring to round up on sells.
#         for stop_price ,reverse
#     ---- narrow the sacle of limits and stop
#     :param price:
#     :param prefer_round_down:
#     :param tick_size:
#     :param diff:
#     :return:
#     """
#     # return 小数位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)
#     # 保留tick_size
#     diff *= multiplier
#     # 保留系统精度
#     epsilon = sys.float_info * 10
#     diff = diff - epsilon
#
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded


# def order(self, asset, amount, style =None, order_id=None):
#     """Place an order.
#
#     Parameters
#     ----------
#     asset : zipline.assets.Asset
#         The asset that this order is for.
#     amount : int
#         The amount of shares to order. If ``amount`` is positive, this is
#         the number of shares to buy or cover. If ``amount`` is negative,
#         this is the number of shares to sell or short.
#     style : zipline.finance.execution.ExecutionStyle
#         The execution style for the order.
#     order_id : str, optional
#         The unique identifier for this order.
#
#     Returns
#     -------
#     order_id : str or None
#         The unique identifier for this order, or None if no order was
#         placed.
#
#     Notes
#     -----
#     amount > 0 :: Buy/Cover
#     amount < 0 :: Sell/Short
#     Market order:    order(asset, amount)
#     Limit order:     order(asset, amount, style=LimitOrder(limit_price))
#     Stop order:      order(asset, amount, style=StopOrder(stop_price))
#     StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,
#                            stop_price))
#     """
#     # something could be done with amount to further divide
#     # between buy by share count OR buy shares up to a dollar amount
#     # numeric == share count  AND  "$dollar.cents" == cost amount
#
#     if amount == 0:
#         # Don't bother placing orders for 0 shares.
#         return None
#     elif amount > self.max_shares:
#         # Arbitrary limit of 100 billion (US) shares will never be
#         # exceeded except by a buggy algorithm.
#         raise OverflowError("Can't order more than %d shares" %
#                             self.max_shares)
#
#     is_buy = (amount > 0)
#     order = Order(
#         dt=self.current_dt,
#         asset=asset,
#         amount=amount,
#         stop=style.get_stop_price(is_buy),
#         limit=style.get_limit_price(is_buy),
#         id=order_id
#     )
#
#     self.open_orders[order.asset].append(order)
#     self.orders[order.id] = order


# full_share_count = self.amount * float(ratio)
# new_cost_basics = round(self.cost_basis / float(ratio), 2)
# left_cash = (full_share_count - np.floor(full_share_count)) * new_cost_basics
# self.cost_basis = np.floor(new_cost_basics)
# self.amount = full_share_count
# return left_cash

# def update(self,txn):
#     if self.asset != txn.asset:
#         raise Exception('transaction must be the same with position asset')
#
#     if self.last_sale_dt is None or txn.dt > self.last_sale_dt:
#         self.last_sale_dt = txn.dt
#         self.last_sale_price = txn.price
#
#     total_shares = txn.amount + self.amount
#     if total_shares == 0:
#         # 用于统计transaction是否盈利
#         # self.cost_basis = 0.0
#         position_return = (txn.price - self.cost_basis)/self.cost_basis
#         self.cost_basis = position_return
#     elif total_shares < 0:
#         raise Exception('for present put action is not allowed')
#     else:
#         total_cost = txn.amout * txn.price + self.amount * self.cost_basis
#         new_cost_basis = total_cost / total_shares
#         self.cost_basis = new_cost_basis
#
#     self.amount = total_shares

# def update_position(self,
#                     asset,
#                     amount = None,
#                     last_sale_price = None,
#                     last_sale_date = None,
#                     cost_basis = None):
#     self._dirty_stats = True
#
#     try:
#         position = self.positions[asset]
#     except KeyError:
#         position = Position(asset)
#
#     if amount is not None:
#         position.amount = amount
#     if last_sale_price is not None :
#         position.last_sale_price = last_sale_price
#     if last_sale_date is not None :
#         position.last_sale_date = last_sale_date
#     if cost_basis is not None :
#         position.cost_basis = cost_basis
#
# # 执行
# def execute_transaction(self,txn):
#     self._dirty_stats = True
#
#     asset = txn.asset
#
#     # 新的股票仓位
#     if asset not in self.positions:
#         position = Position(asset)
#     else:
#         position = self.positions[asset]
#
#     position.update(txn)
#
#     if position.amount ==0 :
#         #统计策略的对应的收益率
#         dt = txn.dt
#         algorithm_ret = position.cost_basis
#         asset_origin = position.asset.reason
#         self.record_vars[asset_origin] = {str(dt):algorithm_ret}
#
#         del self.positions[asset]

# def handle_spilts(self,splits):
#     total_leftover_cash = 0
#
#     for asset,ratio in splits.items():
#         if asset in self.positions:
#             position = self.positions[asset]
#             leftover_cash = position.handle_split(asset,ratio)
#             total_leftover_cash += leftover_cash
#     return total_leftover_cash

#将分红或者配股的数据分类存储
# def earn_divdends(self,cash_divdends,stock_divdends):
#     """
#         given a list of divdends where ex_date all the next_trading
#         including divdend and stock_divdend
#     """
#     for cash_divdend in cash_divdends:
#         div_owned = self.positions[cash_divdend['paymen_asset']].earn_divdend(cash_divdend)
#         self._unpaid_divdend[cash_divdend.pay_date].apppend(div_owned)
#
#     for stock_divdend in stock_divdends:
#         div_owned_ = self.positions[stock_divdend['payment_asset']].earn_stock_divdend(stock_divdend)
#         self._unpaid_stock_divdends[stock_divdend.pay_date].append(div_owned_)

# 根据时间执行分红或者配股
# def pay_divdends(self,next_trading_day):
#     """
#         股权登记日，股权除息日（为股权登记日下一个交易日）
#         但是红股的到账时间不一致（制度是固定的）
#         根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
#         投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
#         其中对于分红的时间存在差异
#
#         根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，
#
#         持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
#     """
#     net_cash_payment = 0.0
#
#     # cash divdend
#     try:
#         payments = self._unpaid_divdend[next_trading_day]
#         del self._unpaid_divdend[next_trading_day]
#     except KeyError:
#         payments = []
#
#     for payment in payments:
#         net_cash_payment += payment['cash_amount']
#
#     #stock divdend
#     try:
#         stock_payments = self._unpaid_stock_divdends[next_trading_day]
#     except KeyError:
#         stock_payments = []
#
#     for stock_payment in stock_payments:
#         payment_asset = stock_payment['payment_asset']
#         share_amount = stock_payment['share_amount']
#         if payment_asset in self.positions:
#             position = self.positions[payment_asset]
#         else:
#             position = self.positions[payment_asset] = Position(payment_asset)
#         position.amount  += share_amount
#     return net_cash_payment

# def calculate_position_tracker_stats(positions,stats):
#     """
#         stats ---- PositionStats
#     """
#     longs_count = 0
#     long_exposure = 0
#     shorts_count = 0
#     short_exposure = 0
#
#     for outer_position in positions.values():
#         position = outer_position.inner_position
#         #daily更新价格
#         exposure = position.amount * position.last_sale_price
#         if exposure > 0:
#             longs_count += 1
#             long_exposure += exposure
#         elif exposure < 0:
#             shorts_count +=1
#             short_exposure += exposure
#     #
#     net_exposure = long_exposure + short_exposure
#     gross_exposure = long_exposure - short_exposure
#
#     stats.gross_exposure = gross_exposure
#     stats.long_exposure = long_exposure
#     stats.longs_count = longs_count
#     stats.net_exposure = net_exposure
#     stats.short_exposure = short_exposure
#     stats.shorts_count = shorts_count

# def process_transaction(self,transaction):
#     position = self.position_tracker.positions[asset]
#     amount = position.amount
#     left_amount = amount + transaction.amount
#     if left_amount == 0:
#         self._cash_flow( - self.commission.calculate(transaction))
#         del self._payout_last_sale_price[asset]
#     elif left_amount < 0:
#         raise Exception('禁止融券卖出')
#     # calculate cash
#     self._cash_flow( - transaction.amount * transaction.price)
#     #execute transaction
#     self.position_tracker.execute_transaction(transaction)
#     transaction_dict = transaction.to_dict()
#     self._processed_transaction[transaction.dt].append(transaction_dict)

# def process_commission(self,commission):
#     asset = commission['asset']
#     cost = commission['cost']
#
#     self.position_tracker.handle_commission(asset,cost)
#     self._cash_flow(-cost)

# def process_split(self,splits):
#     """
#         splits --- (asset,ratio)
#     :param splits:
#     :return:
#     """
#     leftover_cash = self.position_tracker.handle_spilts(splits)
#     if leftover_cash > 0 :
#         self._cash_flow(leftover_cash)
#
# def process_divdends(self,next_session,adjustment_reader):
#     """
#         基于时间、仓位获取对应的现金分红、股票分红
#     """
#     position_tracker = self.position_tracker
#     #针对字典 --- set return keys
#     held_sids = set(position_tracker.positions)
#     if held_sids:
#         cash_divdend = adjustment_reader.get_dividends_with_ex_date(
#             held_sids,
#             next_session,
#         )
#         stock_dividends = (
#             adjustment_reader.get_stock_dividends_with_ex_date(
#                 held_sids,
#                 next_session,
#             )
#         )
#     #添加
#     position_tracker.earn_divdends(
#         cash_divdend,stock_dividends
#     )
#     #基于session --- pay_date 处理
#     self._cash_flow(
#         position_tracker.pay_divdends(next_session)
#     )
# self.record_vars
# def update_portfolio(self):
#     """
#         force a computation of the current portfolio
#         portofolio 保留最新
#     """
#     if not self._dirty_portfolio:
#         return
#
#     portfolio = self._portfolio
#     pt = self.position_tracker
#
#     portfolio.positions = pt.get_positions()
#     #计算positioin stats --- sync_last_sale_price
#     position_stats = pt.stats
#
#     portfolio.positions_value = position_value = (
#         position_stats.net_value
#     )
#
#     portfolio.positions_exposure = position_stats.net_exposure
#     self._cash_flow(self._get_payout_total(pt.positions))
#
#     # portfolio_value 初始化capital_value
#     start_value = portfolio.portfolio_value
#     portfolio.portfolio_value = end_value = portfolio.cash + position_value
#
#     # daily 每天账户净值波动
#     pnl = end_value - start_value
#     if start_value !=0 :
#         returns = pnl/start_value
#     else:
#         returns = 0.0
#
#     #pnl --- 投资收益
#     portfolio.pnl += pnl
#     # 每天的区间收益率 --- 递归方式
#     portfolio.returns = (
#         (1+portfolio.returns) *
#         (1+returns) - 1
#     )
#     self._dirty_portfolio = False
# for asset, old_price in payout_last_sale_prices.items():
#     position = positions[asset]
#     payout_last_sale_prices[asset] = price = position.last_sale_price
#     amount = position.amount
#     total += calculate_payout(
#         amount,
#         old_price,
#         price,
#         asset.price_multiplier,
#     )
# return total
import numpy as np
from scipy.optimize import fsolve
#
# a = [15.3,14.7,14.9,14.01,15.2,16.7,16.9]
# print(np.std(a))
# b = np.std(a)
#
#
# def func(paramlist):
#
#     a,b =paramlist[0],paramlist[1]
#     return [a / (a+b) - 0.0476,
#             (a*b) /((a+b+1) * (a+b) ** 2) - 0.0021]
# c1,c2=fsolve(func,[0,0])
# print(c1,c2)
# e = c1 / (c1+c2)


# a = 10
# b = 200
# e = a/(a + b)
# s = (a*b) /((a+b+1) * (a+b) ** 2)
# print(e,s)
# pct = [0.02,-0.03,0.04,0.05,0.08,-0.06,0.07]
# print(np.std(pct))
# def override_account_fields(self,
#                             settled_cash=not_overridden,
#                             total_positions_values=not_overridden,
#                             total_position_exposure=not_overridden,
#                             cushion=not_overridden,
#                             gross_leverage=not_overridden,
#                             net_leverage=not_overridden,
#                             ):
#     # locals ---函数内部的参数
#     self._account_overrides = kwargs = {k: v for k, v in locals().items() if v is not not_overridden}
#     del kwargs['self']
# for k, v in self._account_overrides:
#     setattr(account, k, v)
from itertools import product

# p = {'a':[1,2,3],'b':[4,5,6],'c':[7,8,9]}
# items = p.items()
#
# keys, values = zip(*items)
# print(keys)
# print(values)
# #product --- 每个列表里面取一个元素
# for v in product(*values):
#     params = dict(zip(keys, v))
#     print(params)
#
# from toolz import concatv
#
# list(concatv([], ["a"], ["b", "c"]))
# #['a', 'b', 'c']
#
# #代码高亮
# try:
#     from pygments import highlight
#     from pygments.lexers import PythonLexer
#     from pygments.formatters import TerminalFormatter
#     PYGMENTS = True
# except ImportError:
#     PYGMENTS = False
#
# """
#     将不同的算法通过串行或者并行方式形成算法工厂 ，筛选过滤最终得出目标目标标的组合
#     串行：
#         1、串行特征工厂借鉴zipline或者scikit_learn Pipeline
#         2、串行理论基础：现行的策略大多数基于串行，比如多头排列、空头排列、行业龙头战法、统计指标筛选
#         3、缺点：确定存在主观去判断特征顺序，前提对于市场有一套自己的认识以及分析方法
#     并行：
#         1、并行的理论基础借鉴交集理论
#         2、基于结果反向分类strategy
#     难点：
#         不同算法的权重分配
#     input : stategies ,output : list or tuple of filtered assets
#
#             pipe of strategy to fit targeted asset
#     Parameters
#     -----------
#     steps :list
#         List of strategies
#         wgts: List,str or list , default : 'average'
#     wgts: List
#         List of (name,weight) tuples that allocate the weight of steps means the
#         importance, average wgts avoids the unbalance of steps
#     memory : joblib.Memory interface used to cache the fitted transformers of
#         the Pipeline. By default,no caching is performed. If a string is given,
#         it is the path to the caching directory. Enabling caching triggers a clone
#         of the transformers before fitting.Caching the transformers is advantageous
#         when fitting is time consuming.
#
#     This estimator applies a list of transformer objects in parallel to the
#     input data, then concatenates the results. This is useful to combine
#     several feature extraction mechanisms into a single transformer.
#
#     Parameters
#     ----------
#     transformer_list : List of transformer objects to be applied to the data
#     n_jobs : int --- Number of jobs to run in parallel,
#             -1 means using all processors.`
#     allocation: str(default=average) ,dict , callable
# """
#
# class Ump(object):
#     """
#         裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
#         关于仲裁逻辑：
#             普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
#             symbol投出反对票，一旦一个因子投出反对票，即筛出序列
#     """
#
#     def __init__(self, poll_workers, thres=0.8):
#         super()._validate_steps(poll_workers)
#         self.voters = poll_workers
#         self._poll_picker = dict()
#         self.threshold = thres
#
#     def _set_params(self, **params):
#         for pname, pval in params.items():
#             self._poll_picker[pname] = pval
#
#     def poll_pick(self, res, v):
#         """
#            vote for feature and quantity the vote action
#            simple poll_pick --- calculate rank pct
#            return bool
#         """
#         formatting = pd.Series(range(1, len(res) + 1), index=res)
#         pct_rank = formatting.rank(pct=True)
#         polling = True if pct_rank[v] > self.thres else False
#         return polling
#
#     def _fit(self, worker, target):
#         '''因子对象针对每一个交易目标的投票结果'''
#         picker = super()._load_from_name(worker)
#         fit_result = picker(self._poll_picker[worker]).fit()
#         poll = self.poll_pick(fit_result, target)
#         return poll
#
#     def decision_function(self, asset):
#         vote_poll = dict()
#         for picker in self.voters:
#             vote_poll.update({picker: self._fit(picker, asset)})
#         decision = np.sum(list(vote_poll.values)) / len(vote_poll)
#         return decision
#
# class MIFeature(ABC):
#     """
#         strategy composed of features which are logically arranged
#         input : feature_list
#         return : asset_list
#         param : _n_field --- all needed field ,_max_window --- upper window along the window args
#         core_part : _domain --- logically combine all features
#     """
#     _n_fields  = []
#     _max_window = []
#     _feature_params = {}
#
#     def _load_features(self,name):
#         try:
#             feature_class = importlib.__import__(name, 'algorithm.features')
#         except:
#             raise ValueError('%s feature not implemented'%name)
#         return feature_class
#
#     def _verify_params(self,params):
#         if isinstance(params,dict):
#             for name,p in params:
#                 feature = self._load_features(name)
#                 if hasattr(feature,'_n_fields') and feature._n_fields != p['fields']:
#                     raise ValueError('fields must be same with feature : %s'%name)
#                 if feature.windowed and p['window'] is None:
#                     raise ValueError('window of feature  is not None : %s'%name)
#                 if feature._pairwise and not isinstance(p['window'],(tuple,list)):
#                     raise ValueError('when pairwise is True ,the length of window must be two')
#                 if hasattr(feature,'_triple') and not isinstance(p['window'],dict):
#                     raise ValueError('triple means three window , it specify macd --- fast,slow,period')
#         else:
#             raise TypeError('params must be dict type')
#
#     def _set_params(self,params):
#         self._verify_params(params)
#         return params
#
#
#     def _eval_feature(self,raw,name,p:dict):
#         """
#             特征分为主体、部分，其中部分特征只是作为主体特征的部分逻辑
#         """
#         feature_class = self._load_features(name)
#         if 'field' in p.keys():
#             print('filed exists spceify this feature should be initialized')
#             if 'window' in p.key():
#                 result = feature_class.calc_feature(raw[p['field']],p['window'])
#             else:
#                 result = feature_class.calc_feature(raw['field'])
#         else:
#             print('field not exists spceify this feature is just  a middle process used by outer faeture function')
#             result = None
#         return result
#
#
#     def _fit_main_features(self,raw):
#         """
#             计算每个标的的所有特征
#         """
#         filter_nan = {}
#         for name in self._n_features:
#             res = self._eval_feature(raw,name,self._feature_params[name])
#             if res:
#                 filter_nan.update({name:res})
#         return filter_nan
#
#
#     def _execute_main(self, trade_date,stock_list):
#         feature_res = {}
#         for code in stock_list:
#             event = Event(trade_date,code)
#             req = GateReq(event, field=self._n_fields, window=self._max_window)
#             raw = feed.addBars(req)
#             res = self._fit_main_features(raw)
#             feature_res.update({code:res})
#         return feature_res
#
#     @abstractmethod
#     def _domain(self,input):
#         """
#             MIFeature（构建有特征组成的接口类），特征按照一定逻辑组合处理为策略
#             实现： 逻辑组合抽象为不同的特征的逻辑运算，具体还是基于不同的特征的运行结果
#         """
#         NotImplemented
#
#
#     def run(self,trade_dt,stock_list:list) -> list:
#         exec_info= self._execute_main(trade_dt,stock_list)
#         filter_order = self._domain(exec_info)
#         return filter_order
#
#
# class MyStrategy(MIFeature):
#     """
#         以MyStrategy为例进行实现
#     """
#
#     _n_features = ['DMA','Reg']
#
#     def __init__(self,params):
#         self._feature_params = super()._set_params(params)
#         self._n_fields = [ v['field'] for k,v in params.items() if 'field' in v.keys()]
#         self._max_window = [ v['window'] for k,v in params.items() if 'window' in v.keys()].max()
#
#     def __enter__(self):
#         return self
#
#     def _domain(self,input):
#         """
#             策略核心逻辑： DMA --- 短期MA大于的长期MA概率超过80%以及收盘价处于最高价与最低价的形成夹角1/2位以上，则asset有效
#             return ranked_list
#         """
#         df = pd.DataFrame.from_dict(input)
#         result = df.T
#         hit_rate = result['DMA'].applymap(lambda x : len(x[x>0])/len(x) > 0.75)
#         reg = result['Reg'].map(lambda x : x > 0.6)
#         # union = set(reg.index) & set(hit_rate.index)
#         input = (pd.DataFrame([hit_rate,reg])).T
#         union = BaseScorer().calc_feature(input)
#         return union
#
#     def __exit__(self,exc_type,exc_val,exc_tb):
#         """
#             exc_type,exc_value,exc_tb(traceback), 当with 后面语句执行错误输出
#         """
#         if exc_val :
#             print('strategy fails to complete')
#         else:
#             print('successfully process')
#
#
# class UnionEngine(object):
#     """
#         组合不同算法---策略
#         返回 --- Order对象
#         initialize
#         handle_data
#         before_trading_start
#         1.判断已经持仓是否卖出
#         2.基于持仓限制确定是否执行买入操作
#     """
#     def __init__(self,algo_mappings,data_portal,blotter,assign_policy):
#         self.data_portal = data_portal
#         self.postion_allocation = assign_policy
#         self.blotter = blotter
#         self.loaders = [self.get_loader_class(key,args) for key,args in algo_mappings.items()]
#
#     @staticmethod
#     def get_loader_class(key,args):
#         """
#         :param key: algo_name or algo_path
#         :param args: algo_params
#         :return: dict -- __name__ : instance
#         """
#
#     # @lru_cache(maxsize=32)
#     def compute_withdraw(self,dt):
#         def run(ins):
#             result = ins.before_trading_start(dt)
#             return result
#
#         with Pool(processes = len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run,instance)
#                             for instance in self.loaders.values]
#         return exit_assets
#
#     # @lru_cache(maxsize=32)
#     def compute_algorithm(self,dt,metrics_tracker):
#         unprocessed_loaders = self.tracker_algorithm(metrics_tracker)
#         def run(algo):
#             ins = self.loaders[algo]
#             result = ins.initialize(dt)
#             return result
#
#         with Pool(processes=len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run, algo)
#                            for algo in unprocessed_loaders]
#         return exit_assets
#
#     def tracker_algorithm(self,metrics_tracker):
#         unprocessed_algo = set(self.algorithm_mappings.keys()) - \
#                            set(map(lambda x : x.reason ,metrics_tracker.positions.assets))
#         return unprocessed_algo
#
#     def position_allocation(self):
#         return self.assign_policy.map_allocation(self.tracker_algorithm)
#
#     def _calculate_order_amount(self,asset,dt,total_value):
#         """
#             calculate how many shares to order based on the position managment
#             and price where is assigned to 10% limit in order to carry out order max amount
#         """
#         preclose = self.data_portal.get_preclose(asset,dt)
#         porportion = self.postion_allocation.compute_pos_placing(asset)
#         amount = np.floor(porportion * total_value / (preclose * 1.1))
#         return amount
#
#     def get_payout(self, dt,metrics_tracker):
#         """
#         :param metrics_tracker: to get the position
#         :return: sell_orders
#         """
#         assets_of_exit = self.compute_withdraw(dt)
#         positions = metrics_tracker.positions
#         if assets_of_exit:
#             [self.blotter.order(asset,
#                                 positions[asset].amount)
#                                 for asset in assets_of_exit]
#             cleanup_transactions,additional_commissions = self.blotter.get_transaction(self.data_portal)
#             return cleanup_transactions,additional_commissions
#
#     def get_layout(self,dt,metrics_tracker):
#         assets = self.compute_algorithm(dt,metrics_tracker)
#         avaiable_cash = metrics_tracker.portfolio.cash
#         [self.blotter.order(asset,
#                             self._calculate_order_amount(asset,dt,avaiable_cash))
#                             for asset in assets]
#         transactions,new_commissions = self.blotter.get_transaction(self.data_portal)
#         return transactions,new_commissions
#
#     def _pop_params(cls, kwargs):
#         """
#         Pop entries from the `kwargs` passed to cls.__new__ based on the values
#         in `cls.params`.
#
#         Parameters
#         ----------
#         kwargs : dict
#             The kwargs passed to cls.__new__.
#
#         Returns
#         -------
#         params : list[(str, object)]
#             A list of string, value pairs containing the entries in cls.params.
#
#         Raises
#         ------
#         TypeError
#             Raised if any parameter values are not passed or not hashable.
#         """
#         params = cls.params
#         if not isinstance(params, Mapping):
#             params = {k: NotSpecified for k in params}
#         param_values = []
#         for key, default_value in params.items():
#             try:
#                 value = kwargs.pop(key, default_value)
#                 if value is NotSpecified:
#                     raise KeyError(key)
#
#                 # Check here that the value is hashable so that we fail here
#                 # instead of trying to hash the param values tuple later.
#                 hash(value)
#             except KeyError:
#                 raise TypeError(
#                     "{typename} expected a keyword parameter {name!r}.".format(
#                         typename=cls.__name__,
#                         name=key
#                     )
#                 )
#             except TypeError:
#                 # Value wasn't hashable.
#                 raise TypeError(
#                     "{typename} expected a hashable value for parameter "
#                     "{name!r}, but got {value!r} instead.".format(
#                         typename=cls.__name__,
#                         name=key,
#                         value=value,
#                     )
#                 )
#
#             param_values.append((key, value))
#         return tuple(param_values)
#
#
# class NoHooks(PipelineHooks):
#     """A PipelineHooks that defines no-op methods for all available hooks.
#     """
#     @contextmanager
#     def running_pipeline(self, pipeline, start_date, end_date):
#         yield
#
#     @contextmanager
#     def computing_chunk(self, terms, start_date, end_date):
#         yield
#
#     @contextmanager
#     def loading_terms(self, terms):
#         yield
#
#     @contextmanager
#     def computing_term(self, term):
#         yield

# @contextmanager
# @abstractmethod
# def loading_terms(self, terms):
#     """Contextmanager entered when loading a batch of LoadableTerms.
#
#     Parameters
#     ----------
#     terms : list[zipline.pipeline.LoadableTerm]
#         Terms being loaded.
#     """
#
# @contextmanager
# @abstractmethod
# def computing_term(self, term):
#     """Contextmanager entered when computing a ComputableTerm.
#
#     Parameters
#     ----------
#     terms : zipline.pipeline.ComputableTerm
#         Terms being computed.
#     """
#
# def delegating_hooks_method(method_name):
#     """Factory function for making DelegatingHooks methods.
#     """
#     if method_name in PIPELINE_HOOKS_CONTEXT_MANAGERS:
#         # Generate a contextmanager that enters the context of all child hooks.
#         # wraps --- callable
#         @wraps(getattr(PipelineHooks, method_name))
#         @contextmanager
#         def ctx(self, *args, **kwargs):
#             with ExitStack() as stack:
#                 for hook in self._hooks:
#                     sub_ctx = getattr(hook, method_name)(*args, **kwargs)
#                     stack.enter_context(sub_ctx)
#                 yield stack
#         return ctx
#     else:
#         # Generate a method that calls methods of all child hooks.
#         @wraps(getattr(PipelineHooks, method_name))
#         def method(self, *args, **kwargs):
#             for hook in self._hooks:
#                 sub_method = getattr(hook, method_name)
#                 sub_method(*args, **kwargs)
#
#         return method
#
#
# class DelegatingHooks(PipelineHooks):
#     """A PipelineHooks that delegates to one or more other hooks.
#
#     Parameters
#     ----------
#     hooks : list[implements(PipelineHooks)]
#         Sequence of hooks to delegate to.
#     """
#     def __new__(cls, hooks):
#         if len(hooks) == 0:
#             # OPTIMIZATION: Short-circuit to a NoHooks if we don't have any
#             # sub-hooks.
#             return NoHooks()
#         else:
#             self = super(DelegatingHooks, cls).__new__(cls)
#             self._hooks = hooks
#             return self
#
#     # Implement all interface methods by delegating to corresponding methods on
#     # input hooks. locals --- __dict__ 覆盖原来的方法
#     locals().update({
#         name: delegating_hooks_method(name)
#         # TODO: Expose this publicly on interface.
#         for name in PipelineHooks._signatures
#     })
#
#
# del delegating_hooks_method
#
# class AlgorithmSimulator(object):
#
#     EMISSION_TO_PERF_KEY_MAP = {
#         'minute': 'minute_perf',
#         'daily': 'daily_perf'
#     }
#
#     def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
#                  restrictions, universe_func):
#
#         # ==============
#         # Simulation
#         # Param Setup
#         # ==============
#         self.sim_params = sim_params
#         self.data_portal = data_portal
#         self.restrictions = restrictions
#
#         # ==============
#         # Algo Setup
#         # ==============
#         self.algo = algo
#
#         # ==============
#         # Snapshot Setup
#         # ==============
#
#         # We don't have a datetime for the current snapshot until we
#         # receive a message.
#         self.simulation_dt = None
#
#         self.clock = clock
#
#         self.benchmark_source = benchmark_source
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#         # This object is the way that user algorithms interact with OHLCV data,
#         # fetcher data, and some API methods like `data.can_trade`.
#         self.current_data = self._create_bar_data(universe_func)
#
#     def get_simulation_dt(self):
#         return self.simulation_dt
#
#     #获取日数据，封装为一个API(fetch process flush other api)
#     def _create_bar_data(self, universe_func):
#         return BarData(
#             data_portal=self.data_portal,
#             simulation_dt_func=self.get_simulation_dt,
#             data_frequency=self.sim_params.data_frequency,
#             trading_calendar=self.algo.trading_calendar,
#             restrictions=self.restrictions,
#             universe_func=universe_func
#         )
#
#     def transform(self):
#         """
#         Main generator work loop.
#         """
#         algo = self.algo
#         metrics_tracker = algo.metrics_tracker
#         emission_rate = metrics_tracker.emission_rate
#
#         #生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
#         def every_bar(dt_to_use, current_data=self.current_data,
#                       handle_data=algo.event_manager.handle_data):
#             for capital_change in calculate_minute_capital_changes(dt_to_use):
#                 yield capital_change
#
#             self.simulation_dt = dt_to_use
#             # called every tick (minute or day).
#             algo.on_dt_changed(dt_to_use)
#
#             blotter = algo.blotter
#
#             # handle any transactions and commissions coming out new orders
#             # placed in the last bar
#             new_transactions, new_commissions, closed_orders = \
#                 blotter.get_transactions(current_data)
#
#             blotter.prune_orders(closed_orders)
#
#             for transaction in new_transactions:
#                 metrics_tracker.process_transaction(transaction)
#
#                 # since this order was modified, record it
#                 order = blotter.orders[transaction.order_id]
#                 metrics_tracker.process_order(order)
#
#             for commission in new_commissions:
#                 metrics_tracker.process_commission(commission)
#
#             handle_data(algo, current_data, dt_to_use)
#
#             # grab any new orders from the blotter, then clear the list.
#             # this includes cancelled orders.
#             new_orders = blotter.new_orders
#             blotter.new_orders = []
#
#             # if we have any new orders, record them so that we know
#             # in what perf period they were placed.
#             for new_order in new_orders:
#                 metrics_tracker.process_order(new_order)
#
#         def once_a_day(midnight_dt, current_data=self.current_data,
#                        data_portal=self.data_portal):
#             # process any capital changes that came overnight
#             for capital_change in algo.calculate_capital_changes(
#                     midnight_dt, emission_rate=emission_rate,
#                     is_interday=True):
#                 yield capital_change
#
#             # set all the timestamps
#             self.simulation_dt = midnight_dt
#             algo.on_dt_changed(midnight_dt)
#
#             metrics_tracker.handle_market_open(
#                 midnight_dt,
#                 algo.data_portal,
#             )
#
#             # handle any splits that impact any positions or any open orders.
#             assets_we_care_about = (
#                 viewkeys(metrics_tracker.positions) |
#                 viewkeys(algo.blotter.open_orders)
#             )
#
#             if assets_we_care_about:
#                 splits = data_portal.get_splits(assets_we_care_about,
#                                                 midnight_dt)
#                 if splits:
#                     algo.blotter.process_splits(splits)
#                     metrics_tracker.handle_splits(splits)
#
#         def on_exit():
#             # Remove references to algo, data portal, et al to break cycles
#             # and ensure deterministic cleanup of these objects when the
#             # simulation finishes.
#             self.algo = None
#             self.benchmark_source = self.current_data = self.data_portal = None
#
#         with ExitStack() as stack:
#             """
#             由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
#             这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
#             enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
#             callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
#             """
#             stack.callback(on_exit)
#             stack.enter_context(self.processor)
#             stack.enter_context(ZiplineAPI(self.algo))
#
#             if algo.data_frequency == 'minute':
#                 def execute_order_cancellation_policy():
#                     algo.blotter.execute_cancel_policy(SESSION_END)
#
#                 def calculate_minute_capital_changes(dt):
#                     # process any capital changes that came between the last
#                     # and current minutes
#                     return algo.calculate_capital_changes(
#                         dt, emission_rate=emission_rate, is_interday=False)
#             else:
#                 def execute_order_cancellation_policy():
#                     pass
#
#                 def calculate_minute_capital_changes(dt):
#                     return []
#
#             for dt, action in self.clock:
#                 if action == BAR:
#                     for capital_change_packet in every_bar(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_START:
#                     for capital_change_packet in once_a_day(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_END:
#                     # End of the session.
#                     positions = metrics_tracker.positions
#                     position_assets = algo.asset_finder.retrieve_all(positions)
#                     self._cleanup_expired_assets(dt, position_assets)
#
#                     execute_order_cancellation_policy()
#                     algo.validate_account_controls()
#
#                     yield self._get_daily_message(dt, algo, metrics_tracker)
#                 elif action == BEFORE_TRADING_START_BAR:
#                     self.simulation_dt = dt
#                     algo.on_dt_changed(dt)
#                     algo.before_trading_start(self.current_data)
#                 elif action == MINUTE_END:
#                     minute_msg = self._get_minute_message(
#                         dt,
#                         algo,
#                         metrics_tracker,
#                     )
#
#                     yield minute_msg
#
#             risk_message = metrics_tracker.handle_simulation_end(
#                 self.data_portal,
#             )
#             yield risk_message
#
#     def _cleanup_expired_assets(self, dt, position_assets):
#         """
#         Clear out any assets that have expired before starting a new sim day.
#
#         Performs two functions:
#
#         1. Finds all assets for which we have open orders and clears any
#            orders whose assets are on or after their auto_close_date.
#
#         2. Finds all assets for which we have positions and generates
#            close_position events for any assets that have reached their
#            auto_close_date.
#         """
#         algo = self.algo
#
#         def past_auto_close_date(asset):
#             acd = asset.auto_close_date
#             return acd is not None and acd <= dt
#
#         # Remove positions in any sids that have reached their auto_close date.
#         assets_to_clear = \
#             [asset for asset in position_assets if past_auto_close_date(asset)]
#         metrics_tracker = algo.metrics_tracker
#         data_portal = self.data_portal
#         for asset in assets_to_clear:
#             metrics_tracker.process_close_position(asset, dt, data_portal)
#
#         # Remove open orders for any sids that have reached their auto close
#         # date. These orders get processed immediately because otherwise they
#         # would not be processed until the first bar of the next day.
#         blotter = algo.blotter
#         assets_to_cancel = [
#             asset for asset in blotter.open_orders
#             if past_auto_close_date(asset)
#         ]
#         for asset in assets_to_cancel:
#             blotter.cancel_all_orders_for_asset(asset)
#
#         # Make a copy here so that we are not modifying the list that is being
#         # iterated over.
#         for order in copy(blotter.new_orders):
#             if order.status == ORDER_STATUS.CANCELLED:
#                 metrics_tracker.process_order(order)
#                 blotter.new_orders.remove(order)
#
#     def _get_daily_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         perf_message = metrics_tracker.handle_market_close(
#             dt,
#             self.data_portal,
#         )
#         perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
#         return perf_message
#
#     def _get_minute_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         rvars = algo.recorded_vars
#
#         minute_message = metrics_tracker.handle_minute_close(
#             dt,
#             self.data_portal,
#         )
#
#         minute_message['minute_perf']['recorded_vars'] = rvars
#         return minute_message
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#
# class PeriodLabel(object):
#     """Backwards compat, please kill me.
#     """
#     def start_of_session(self, ledger, session, data_portal):
#         self._label = session.strftime('%Y-%m')
#
#     def end_of_bar(self, packet, *args):
#         packet['cumulative_risk_metrics']['period_label'] = self._label
#
#     end_of_session = end_of_bar
#
#
# class _ConstantCumulativeRiskMetric(object):
#     """A metrics which does not change, ever.
#
#     Notes
#     -----
#     This exists to maintain the existing structure of the perf packets. We
#     should kill this as soon as possible.
#     """
#     def __init__(self, field, value):
#         self._field = field
#         self._value = value
#
#     def start_of_session(self, packet,*args):
#         packet['cumulative_risk_metrics'][self._field] = self._value
#
#     def end_of_session(self, packet, *args):
#         packet['cumulative_risk_metrics'][self._field] = self._value


# If you are adding new attributes, don't update this set. This method
# is deprecated to normal attribute access so we don't want to encourage
# new usages.
# __getitem__ = _deprecated_getitem_method(
#     'portfolio', {
#         'capital_used',
#         'starting_cash',
#         'portfolio_value',
#         'pnl',
#         'returns',
#         'cash',
#         'positions',
#         'start_date',
#         'positions_value',
#     },
# )

#toolz.itertoolz.groupby(key, seq)
from dateutil.relativedelta import relativedelta
import datetime , pandas as pd
#
# start_session = datetime.datetime.strptime('2010-01-31','%Y-%m-%d')
# end_session = datetime.datetime.strptime('2012-01-31','%Y-%m-%d')
#
# print(start_session,end_session)
#
# # end = end_session.replace(day=1) + relativedelta(months=1)
# end = end_session
# print(end)
#
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end,
#     freq='M',
#     tz='utc',
#     closed = 'left'
# )
# print('months',months.size)
# print(type(months),months)
# months.iloc[-1] = 'test'
# period = months[0].to_period(freq='%dM' % 3)
# print(months[::3])
# print('period',period.end_date)


# for period_timestamp in months:
#     period = period_timestamp.to_period(freq='%dM' % months_per)

# # 下个月第一天
# end = end_session.replace(day=1) + relativedelta(months=1)
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end - datetime.timedelta(days=1),
#     freq='M',
#     tz='utc',
# )
# 分析指标:
# 策略共执行{}个交易日 策略资金利用率比例  策略买入成交比例 平均获利期望 平均亏损期望
# 策略持股天数平均值,策略持股天数中位数,策略期望收益,策略期望亏损,前后两两生效交易时间相减,
# 计算平均生效间隔时间,计算cost各种统计度量值,计算资金对应的成交比例
# from sys import float_info
#
# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#     Asymmetric rounding function for adjusting prices to the specified number
#     of places in a way that "improves" the price. For limit prices, this means
#     preferring to round down on buys and preferring to round up on sells.
#     For stop prices, it means the reverse.
#
#     If prefer_round_down == True:
#         When .05 below to .95 above a specified decimal place, use it.
#     If prefer_round_down == False:
#         When .95 below to .05 above a specified decimal place, use it.
#
#     In math-speak:
#     If prefer_round_down: [<X-1>.0095, X.0195) -> round to X.01.
#     If not prefer_round_down: (<X-1>.0005, X.0105] -> round to X.01.
#     """
#     # 返回位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)  # adjust diff to precision of tick size
#     diff *= multiplier  # adjust diff to value of tick_size
#
#     # Subtracting an epsilon from diff to enforce the open-ness of the upper
#     # bound on buys and the lower bound on sells.  Using the actual system
#     # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
#     epsilon = float_info.epsilon * 10
#     diff = diff - epsilon
#
#     # relies on rounding half away from zero, unlike numpy's bankers' rounding
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded
#
#
# # 生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
# def every_bar(dt_to_use, current_data=self.current_data,
#               handle_data=algo.event_manager.handle_data):
#     for capital_change in calculate_minute_capital_changes(dt_to_use):
#         yield capital_change
#
#     self.simulation_dt = dt_to_use
#     # called every tick (minute or day).
#     algo.on_dt_changed(dt_to_use)
#
#     blotter = algo.blotter
#
#     # handle any transactions and commissions coming out new orders
#     # placed in the last bar
#     new_transactions, new_commissions, closed_orders = \
#         blotter.get_transactions(current_data)
#
#     blotter.prune_orders(closed_orders)
#
#     for transaction in new_transactions:
#         metrics_tracker.process_transaction(transaction)
#
#         # since this order was modified, record it
#         order = blotter.orders[transaction.order_id]
#         metrics_tracker.process_order(order)
#
#     for commission in new_commissions:
#         metrics_tracker.process_commission(commission)
#
#     handle_data(algo, current_data, dt_to_use)
#
#     # grab any new orders from the blotter, then clear the list.
#     # this includes cancelled orders.
#     new_orders = blotter.new_orders
#     blotter.new_orders = []
#
#     # if we have any new orders, record them so that we know
#     # in what perf period they were placed.
#     for new_order in new_orders:
#         metrics_tracker.process_order(new_order)
#
# def once_a_day(midnight_dt, current_data=self.current_data,
#                data_portal=self.data_portal):
#     # process any capital changes that came overnight
#     for capital_change in algo.calculate_capital_changes(
#             midnight_dt, emission_rate=emission_rate,
#             is_interday=True):
#         yield capital_change
#
#     # set all the timestamps
#     self.simulation_dt = midnight_dt
#     algo.on_dt_changed(midnight_dt)
#
#     metrics_tracker.handle_market_open(
#         midnight_dt,
#         algo.data_portal,
#     )
#
#     # handle any splits that impact any positions or any open orders.
#     assets_we_care_about = (
#         viewkeys(metrics_tracker.positions) |
#         viewkeys(algo.blotter.open_orders)
#     )
#
#     if assets_we_care_about:
#         splits = data_portal.get_splits(assets_we_care_about,
#                                         midnight_dt)
#         if splits:
#             algo.blotter.process_splits(splits)
#             metrics_tracker.handle_splits(splits)
#
