# -*- coding : uft-8 -*-

import pandas as pd,warnings
from types import MappingProxyType as mappingproxy
# 返回一个动态映射视图

class Positions(dict):
    """
        a dict_object containing the algorithm's current positions
    """
    #类似于 defaultdict
    def __missing__(self, key):
        if isinstance(key,Asset):
            return Position(InnerPosition(key))
        elif isinstance(key,int):
            warnings.warn('referencing positions by integer is deprecated use an asset instead')
        else:
            warnings.warn('position lookup expected a value of type asset but got{0} instead').format(type(key).__name__)

#订单类
import uuid
from enum import Enum

#订单类型
import numpy as np ,sys
from abc import ABC , abstractmethod

from collections import defaultdict

#sentinel f返回目标的具体信息， 代码文件名以及行号 ，基于_getframe
from textwrap import dedent
import sys

class _Sentinel(object):
    """
        base sentinel is used when you only care to check for object identity
    """
    __slots__ = ('__wreakref__',)

def is_sentinel(obj):
    return isinstance(obj,_Sentinel)

#主体
def sentinel(name,doc= None):
    try:
        value = sentinel._cache[name]
    except KeyError:
        pass
    else:
        if doc == value.__doc__:
            return value
        raise ValueError(dedent(
            """new sentinel conflicts with existing sentinel """
        ))

    try:
        #default is 0 ,the top of stack
        frame = sys._getframe(1)
    except ValueError:
        frame = None

    if frame is None:
        created_at = '<unkown>'
    else:
        #返回基本信息
        created_at = '%s:%s'(frame.f_code.co_filename,frame.lineno)

    @object.__new__
    class Sentinel(_Sentinel):

        __doc__ = doc
        __name__ = name

        def __new__(cls):
            raise TypeError('cannot create %r instance'%name)

        def __repr__(self):
            return 'sentinel(%r)'%name

        def __reduce__(self):
            return sentinel , (name,doc)

        def __deepcoy__(self,_memo):
            return self

        def __copy__(self):
            return self

    cls = type(Sentinel)
    try:
        cls.__module__ = frame.f_globals['__name__']
    except (AttributeError,KeyError):
        cls.__module__ = None

    sentinel._cache[name] = Sentinel
    return Sentinel

ExpiredCachedObject = sentinel('ExpiredCachedObject')
AlwaysExpired = sentinel('AlwaysExpired')

#关于dataframe  working_file working_dir cache 主要保存为文件
# python collections 容器
from collections import MutableMapping
from tempfile import mkdtemp , mkstemp,mktemp ,NamedTemporaryFile
import pickle,os,shutil,errno

class dataframe_cache(MutableMapping):
    """
        dataframe_cache is a mutable mapping from string to pandas dataframe objects
        this object may be used as a context manager to delete the cache directory on exit
        __getitem__ __setitem__ __delitem__ 方法 MutableMapping
    """
    def __init__(self,
                 path = None,
                 lock = None,
                 clean_on_failure = True,
                 serialization = 'msgpack'):
        #create directory
        self.path = path if path is not None else mkdtemp
        #nop_context 上下文为pass
        self.lock = lock if lock is not None else nop_context
        self.clearn_on_failure = clean_on_failure

        if serialization == 'msgpack':
            self.serialization = pd.DataFrame.to_msgpack
            self.deserialize = pd.read_msgpack
            self.protocol = None
        else:
            s = serialization.split(':',1)
            if s[0] != 'pickle':
                raise ValueError(
                    "serialization must be either 'msgpack' or 'pickle[:n]'"
                )
            self._protocol = int(s[1]) if len(s) == 2 else None
            self.serialize = self._serialize_pickle
            # self.deserialize = ( pickle.load if PY2 else partial(pickle.load,encoding = 'latin-1'))

    def _serialize_pickle(self,df,path):
        with open(path,'wb') as f:
            pickle.dump(df,f,protocol = self.protocol)

    def _key_path(self,key):
        return os.path.join(self.path,key)
    #上下文
    def  __enter__(self):
        return self

    # type value traceback
    def __exit__(self,type_,val,tb):
        if not (self.clearn_on_failure) or val is None :
            return

        with self.lock:
            shutil.rmtree(self.path)

    def __getitem__(self,key):
        if key == slice(None):
            return dict(self.items())

        with self.lock:
            try:
                with open(self._key_path(key),'rb') as f:
                    return self.deserialize(f)
            except IOError as e:
                if e.errno != errno.ENOENT:
                    raise
                raise KeyError

    def __setitem__(self,key,value):
        with self.lock:
            self.serialization(value,self._key_path(key))

    def __delitem__(self,key):
        try:
            os.remove(self._key_path(key))
        except IOError as e:
            if e.errno == errno.ENOENT:
                raise KeyError
            raise

    def __iter__(self):
        #目录下的子文件
        return iter(os.listdir(self.path))

    def __len__(self):
        return len(os.listdir(self.path))

    def __repr__(self):
        return '<%s : keys = {%s}>'%(
            type(self).__name__,
            ','.join(map(repr,sorted(self)))
        )

# move file to location
class working_file(object):
    """
        a context manager for managing a temporary file that will be moved to a non-temporary location
    """
    def __init__(self,final_path,*args,**kwargs):
        # NamedTemporaryFile has a visble name can be retrieved from the name attribute ,delete --- True means delete after close
        self._tmpfile = NamedTemporaryFile(delete = False , *args , **kwargs)
        self._final_path = final_path

    @property
    def path(self):
        return self._tmpfile.name

    def _commit(self):
        shutil.move(self.path,self._final_path)

    def __enter__(self):
        self._tmpfile.__enter__()
        return self

    # *exec_info -- type_,val , traceback
    def __exit__(self,*exec_info):
        self._tmpfile.__exit__(*exec_info)
        if exec_info[0] is None:
            self._commit()

from distutils import dir_util

class working_dir(object):
    """
        move the directory to  a non_temporary location
    """
    def __init__(self,final_path,*args,**kwargs):
        self.path = mkdtemp()
        self._final_path = final_path

    def get_path(self,*path_parts):

        return os.path.join(self.path,*path_parts)

    def ensure_dir(self,*path_parts):
        """
            ensure the subdirectory of the working directory
        :param path_parts:
        :return:
        """
        path = self.get_path(*path_parts)
        ensure_directory(path)

    def _commit(self):
        """
            sync the temporary directory to the final path
        :return:
        """
        dir_util.copy_tree(self.path,self._final_path)

    def __enter__(self):
        return self

    def __exit__(self,*exec_info):
        if exec_info[0] is None:
            self._commit()
        shutil.rmtree(self.path)

def ensure_directory(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        # directory path 存在
        if exc.errno == EEXIST and os.path.isdir(path):
            return
        raise

import heapq

def _decorate_source(source):
    for message in source:
        yield ((message.dt,message.source_id),message)

# def date_sorted_source(source):
#
#     sorted_stream = heapq.merge(*_decorate_source(s) for s in source)
#
#     for _,message in sorted_stream:
#         yield message


#trading_calendar
from datetime import datetime
from dateutil import rrule
import pytz
from functools import partial

def canonicalize_datetime(dt):
    """
        strip out year month day
    :param dt:
    :return:
    """
    return datetime(dt.year,dt.month,dt.day,tzinfo = pytz.utc)

def get_non_trading_days(start,end):

    pass

def get_open_and_close(day,early_closes):
    """
        返回每天的交易时间
    :param day:
    :return:
    """
    pass

def get_open_and_closes(trading_days,early_close,get_open_and_close):
    open_and_closes = pd.DataFrame(index = trading_days,columns = ('market_open','market_close'))
    get_o_and_c = partial(get_open_and_close,early_closes = early_close)
    get_open_and_closes['market_close'],get_open_and_closes['market_close'] = \
    zip(*open_and_closes.index.map(get_open_and_closes()))
    return open_and_closes

asset_type = ['stock','sci','etf','convertible']
class OrderStatus(Enum):

    OPEN = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4
    HELD = 5

import math

# 交易成本 分为订单如果有交易成本，考虑是否增加额外成本 ； 如果没有交易成本则需要计算
class Commission(ABC):

    @abstractmethod
    def calculate(self):
        raise NotImplementedError


class ExecutionStyle(ABC):
    """
        base class for order execution style
    """
    _exchange = None

    @abstractmethod
    def get_limit_price(self, is_buy):
        raise NotImplementedError

    @abstractmethod
    def get_stop_price(self, is_buy):
        raise NotImplementedError

    @property
    def exchange(self):
        return self._exchange


class MarketOrder(ExecutionStyle):

    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return None


class LimitOrder(ExecutionStyle):
    """
        limit price --- maximum price for buys or minimum price for sells
    """

    def __init__(self, limit_price, asset=None, exchange=None):
        check_stoplimit_prices(limit_price, 'limit')

        self.limit_price = limit_price
        self._exchange = exchange
        self.asset = asset

    def get_limit_price(self, is_buy):
        return asymmetric_round_price(self.limit_price, is_buy,
                                      tick_size=(0.01 if self.asset is None else self.asset.tick_size))

    def get_stop_price(self, _is_buy):
        return None


class StopOrder(ExecutionStyle):
    """
        stop_price ---- for sells the order will be placed if market price falls below this value .
        for buys ,the order will be placed if market price rise above this value.
    """

    def __init__(self, stop_price, asset=None, exchange=None):
        check_stoplimit_prices(stop_price, 'stop')

        self.stop_price = stop_price
        self._exchange = exchange
        self.asset = asset

    def get_limit_price(self, is_buy):
        return None

    def get_stop_price(self, is_buy):
        return asymmetric_round_price(
            self.stop_price,
            not is_buy,
            tick_size=(0.01 if self.asset is None else self.asset.tick_size)
        )


class StopLimitOrder(ExecutionStyle):
    """
        price reach a threahold
    """

    def __init__(self, limit_price, stop_price, asset=None, exchange=None):
        check_stoplimit_prices(limit_price, 'limit')
        check_stoplimit_prices(stop_price, 'stop')

        self.limit_price = limit_price
        self.stop_price = stop_price
        self._exchange = exchange
        self.asset = asset

    def get_limit_price(self, is_buy):
        return asymmetric_round_price(
            self.limit_price,
            is_buy,
            tick_size=(0.01 if self.asset is None else self.asset.tick_size)
        )

    def get_stop_price(self, is_buy):
        return asymmetric_round_price(
            self.stop_price,
            not is_buy,
            tick_size=(0.01 if self.asset is None else self.asset.tick_size)
        )


def check_stoplimit_prices(price, label):
    """
    check to make sure the stop/limit prices are reasonable and raise a badorderParameters
    :param price:
    :param label:
    :return:
    """
    try:
        if not np.isfinite(price):
            raise BadOrderParameter('')
    except TypeError:
        raise BadOrderParameter('')
    if price < 0:
        raise BadOrderParameter('negative price')


def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
    """
        for limit_price ,this means preferring to round down on buys and preferring to round up on sells.
        for stop_price ,reverse
    ---- narrow the sacle of limits and stop
    :param price:
    :param prefer_round_down:
    :param tick_size:
    :param diff:
    :return:
    """
    # return 小数位数
    precision = zp_math.number_of_decimal_places(tick_size)
    multiplier = int(tick_size * (10 ** precision))
    diff -= 0.5  # shift the difference down
    diff *= (10 ** -precision)
    # 保留tick_size
    diff *= multiplier
    # 保留系统精度
    epsilon = sys.float_info * 10
    diff = diff - epsilon

    rounded = tick_size * consistent_round(
        (price - (diff if prefer_round_down else -diff)) / tick_size
    )
    if zp_math.tolerant_equals(rounded, 0.0):
        return 0.0
    return rounded


class LiquidityExceded(Exception):
    pass


class SlippageModel(ABC):

    def __init__(self,data_portal):
        self.data_portal = data_portal

    def fill_worse_than_limit_price(self,price,order):
        preclose = self.data_portal.getPreclose(order.asset,order.dt)
        if price > preclose * 1.1 or price < preclose * 0.9 :
            raise ValueError('price must between -10% 至 10% ')

    @abstractmethod
    def process_order(self,order):
        """
            computer the number of shares and price to fill for order in the day
        """
        raise NotImplementedError

    def simulate(self,asset_order):
        # order.check_trigger(price, dt)
        # if not order.triggered:
        #     continue
        execution_price,execution_amount = self.process_order(asset_order)
        if execution_price is not None :
            self.fill_worse_than_limit_price(execution_price, asset_order)
            txn = create_transaction(asset_order,
                                             execution_price,
                                             execution_amount)
            return txn


class NoSlippage(SlippageModel):
    """
        a slippage model where all orders fill immediately and completely at the current time
    """
    def __init__(self,data_portal):
        super(NoSlippage,self).__init__(data_portal)

    def process_order(self,order):
        price = self.data_portal.current(['open','high','low','close'],order.dt).mean()
        return (
            price,
            order.amount
        )

class FixedBasisPointSlippage(SlippageModel):

    """
        basics_points * 0.0001
    """
    def __init__(self,basis_points = 1.0 ):
        super(FixedBasisPointSlippage,self).__init__()
        self.basis_points = basis_points

    def process_order(self,order):
        price = self.data_portal.current(['open','high','low','close'],order.dt).mean()
        return (
                price + price * (self.basis_points * order.direction /100),
                order.amount
        )


class Expired(Exception):
    """
        mark a cacheobject has expired
    """

class CachedObject(object):
    """
        a simple struct for maintaining a cached object with an expiration date
    """
    def __init__(self,value,expires):
        self._value = value
        # 过期时间
        self._expires = expires

    @classmethod
    def expired(cls):
        """
            contructa cachedobject that is expired at any time
        :return:
        """
        return cls(ExpiredCachedObject,expires = AlwaysExpired)

    def unwrap(self,dt):
        """get wrapper value"""
        expires = self._expires
        if expires is AlwaysExpired or expires < dt:
            raise Expired(self._expires)

    def _unsafe_get_value(self):
        """you almost certainly should not use this"""
        return self._value

class ExpiringCache(object):
    """
        a cache of multiple CacheObjects
    """
    def __init__(self,cache = None,cleanup = lambda value_to_clean :None):
        if cache is not None :
            self._cache = cache
        else:
            self._cache = {}

        self.cleanup = cleanup

    def get(self,key,dt):
        try:
            return self._cache[key].unwrap(dt)
        except Expired:
            self.cleanup(self._cache[key]._unsafe_get_value())
            del self._cache[key]
            raise KeyError

    def set(self,key,value,expiration_dt):
        self._cache[key] = CachedObject(value,expiration_dt)

class MarketImpact(SlippageModel):
    """
        base class for slippage models  which compute a simulated price  impact
        new_price = price + MI * price /10000
        MI --- market impact
        MI = eta * sigma  * sqrt(psi)
        psi --- volume traded divided by 2--day adv
    """
    def __init__(self,data_portal,length = 0):
        self.window = length
        super(MarketImpact,self).__init__(data_portal)

    def get_txn_volume(self,order):
        return order.amount

    def _get_window_data(self,dt,asset,window):

        close_history = self.data_portal.kline(asset,'volume',dt,window)
        close_volatity = (close_history / close_history.shift(1) - 1).std(skipna = False)
        volume_history = self.data_portal.kline(asset,'volume',dt,window)
        values = {'volume':volume_history.mean(),
                  'close':close_volatity}
        return values['volume'],values['close']

    def get_simulate_impact(self,mean_volume,price_volatity):
        psi = self.get_txn_volume() / mean_volume
        impacted_price = price_volatity * np.math.sqrt(psi)
        return impacted_price

    def process_order(self,order):
        if order.open_amount == 0:
            return None,None
        mean_vol , stdev = self._get_window_data(order.dt,order.asset,self.window)
        if mean_vol == 0 or pd.isnull(stdev):
            simulated_price = 0
        else:
            simulated_price = self.get_simulate_impact(mean_vol,stdev)
        price = np.mean(self.data_portal.getCurrent(['open','high','low','close'],order.dt))
        impacted_price = price + np.math.copysign(simulated_price,order.amount)
        return impacted_price



def fill_price_worse_than_limit_price(fill_price,order):
    """
        check whether the fill price is worse than the order's limit price
    :param fill_price:
    :param order:
    :return:
    """
    if order.limit:
        if ( order.direction > 0  and fill_price > order.limit) or \
                (order.direction <0 and fill_price < order.limit ):
            return True
    return False


#交易成本
class Rate(object):

    @property
    def fee(self,dt):
        """
        :param dt: 时点
        :return:万分2。5与千3的手续费
        """
        pass

    @fee.setter
    def fee(self,rate):
        pass

class CommissionModel(ABC):

    @abstractmethod
    def calculate(self,order,transaction):
        raise NotImplementedError


class NoCommission(CommissionModel):

    @staticmethod
    def calculate(order,transaction):
        return 0.0

class EquityCommission(CommissionModel):

    def __init__(self,min_commission = 5):
        self.min_cost = min_commission
        self.per_share_rate = Rate()

    def calculate(self,transaction):

        cost = transaction.amount * self.per_share_rate.fee(transaction.dt)
        txn_cost = cost if cost > self.min_cost else self.min_cost
        return txn_cost

#cancel_policy
class CancelPolicy(ABC):

    @abstractmethod
    def should_cancel(self,order):
        pass


class EODCancel(CancelPolicy):
    """
        eod means the day which asset of order withdraw from market or suspend
    """
    def __init__(self,data_portal):
        self.withdraw_assets = data_portal.get_delist_assets()

    def should_cancel(self,order):
        cancel = False
        try:
            delist_dt = self.withdraw_assets[order.asset]
            if delist_dt <= order.dt :
                cancel = True
        except KeyError:
            pass
        finally:
            return cancel


class LiquityCancel(CancelPolicy):
    """
        liquity means  asset of order cannot be traded because of the limit_stop rule or susupend
    """

class NeverCancel(CancelPolicy):

    def should_cancel(self,order):
        return False


class Order(object):

    __slots__ = ['dt','asset','amount','id','filled']

    def __init__(self,dt,asset,amount,id,filled = 0):
        self.dt = dt
        self.asset = asset
        self.amount = amount
        self.id = id
        self.direction = math.copysign(1,self.amount)
        self.filled = filled

    def make_id(self):
        return uuid.uuid4().hex

    def handle_splits(self,ratio):
        self.amount = int(self.amount / ratio)

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def status(self):
        self._status = OrderStatus.OPEN

    @status.setter
    def status(self,status):
        self._status = status

    def cancel(self):
        self.status = OrderStatus.CANCELLED

    def to_dict(self):
        dct = {name : getattr(self.name)
               for name in self.__slots__}
        return dct


class SimulationBlotter(object):
    """
        process order to transaction
    """

    def __init__(self,
                 dt,
                 data_portal,
                 equity_slippage = NoSlippage,
                 equity_commission = NoCommission,
                 cancel_policy  = LiquityCancel):

        self.simulation_dt = dt
        self.data_portal = data_portal
        self.open_orders = defaultdict(list)
        self.slippage_models =  equity_slippage
        self.commission_models =  equity_commission
        self.cancel_policy = cancel_policy

    def _create_order_list(self,orders):
        for order in orders:
            if not self.cancel_policy.should_cancel(order):
                self.open_orders[order.asset].append(order)

    def process_splits(self):
        """
            splits --- {asset : ratio}
        """
        splits = self.data_portal.get_splits(self.simulation_dt)
        for asset,ratio in splits:
            if asset not in self.open_orders:
                continue
            orders_to_modify = self.open_orders[asset]
            for order in orders_to_modify:
                order.handle_splits(ratio)

    def order(self, asset, amount, style =None, order_id=None):
        """Place an order.

        Parameters
        ----------
        asset : zipline.assets.Asset
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        style : zipline.finance.execution.ExecutionStyle
            The execution style for the order.
        order_id : str, optional
            The unique identifier for this order.

        Returns
        -------
        order_id : str or None
            The unique identifier for this order, or None if no order was
            placed.

        Notes
        -----
        amount > 0 :: Buy/Cover
        amount < 0 :: Sell/Short
        Market order:    order(asset, amount)
        Limit order:     order(asset, amount, style=LimitOrder(limit_price))
        Stop order:      order(asset, amount, style=StopOrder(stop_price))
        StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,
                               stop_price))
        """
        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # numeric == share count  AND  "$dollar.cents" == cost amount

        if amount == 0:
            # Don't bother placing orders for 0 shares.
            return None
        elif amount > self.max_shares:
            # Arbitrary limit of 100 billion (US) shares will never be
            # exceeded except by a buggy algorithm.
            raise OverflowError("Can't order more than %d shares" %
                                self.max_shares)

        is_buy = (amount > 0)
        order = Order(
            dt=self.current_dt,
            asset=asset,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy),
            id=order_id
        )

        self.open_orders[order.asset].append(order)
        self.orders[order.id] = order

    def get_transaction(self):
        closed_orders = []
        transactions = []
        commissions =  []

        for order in self.open_orders.values():
            txn = self.slippage_models.simulate(order)
            cost  = self.commission_models.calculate(txn)
            transactions.append(txn)
            commissions.append({
                'asset':order.asset,
                'order':order,
                'cost':cost
            })
            closed_orders.append(order)
        self.prune_orders(closed_orders)
        return transactions , commissions

    def prune_orders(self,closed_orders):
        """
            remove all given orders from the blotter
        :return:
        """
        for order in closed_orders:
            asset = order.asset
            asset_order = self.open_orders[asset]
            try:
                asset_order.remove(asset_order)
            except ValueError:
                continue

        for asset in list(self.open_orders.keys()):
            if len(self.open_orders[asset]) == 0 :
                del self.open_orders[asset]

# 仓位
class Position(object):
    """
        a position held by algorithm
    """
    __slots__ = ('_underlying_position')

    def __init__(self,underlying_position):
        object.__setattr__(self,'_underlying_position', underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position,attr)

    def __setattr__(self, key, value):
        raise AttributeError('cannot mutate position objects')

    @property
    def sid(self):
        return self.asset

    def __repr__(self):
        return 'position(%r)'%{
            k:getattr(self,k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sale_price',
                'laost_sale_date'
            )
        }


class Position(object):
    __slots__ = ['inner_position','protocol_position']

    def __init__(self,
                 asset,
                 amount = 0,
                 cost_basis = 0.0,
                 # last_sale_price 与last_sale_dt 一一对应的
                 last_sale_price = 0.0,
                 last_sale_date = None):
        inner = InnerPosition(
                asset = asset,
            amount = amount,
            cost_basis = cost_basis,
            #标的的最终价格对应时间
            last_sale_price = last_sale_price,
            last_sale_date = last_sale_date
        )
        object.__setattr__(self,'inner_position',inner)
        object.__setattr__(self,'protocol_position',Position(inner))

    def __getattr__(self, item):
        return getattr(self.inner_position,item)

    def __setattr__(self, key, value):
        setattr(self.inner_position,key,value)

    def earn_divdend(self,divdend):
        """
            registered divdend ex_date and pay out on the pay_date (分红）
        :param divdend:
        :return:
        """
        return {
            'payment_asset':divdend.asset,
            'cash_amount':self.amount * divdend.amount
        }

    def earn_stock_divdend(self,stock_divdend):
        """
            register the number of shares at divdend ex_date (送股、配股）
        :param stock_divdend:
        :return:
        """
        return {
            'payment_asset':stock_divdend.payment_asset,
            'share_amount': np.floor(self.amount * float(stock_divdend.ratio))
        }

    def handle_split(self,asset,ratio):
        """
            update the postion by the split ratio and return the fractional share that will be converted into cash (除权）
            零股转为现金 ,重新计算成本,
            国内A股一般不会产生散股
        """
        if self.asset != asset:
            raise Exception('update wrong asset')

        full_share_count = self.amount * float(ratio)
        new_cost_basics = round(self.cost_basis / float(ratio),2)
        left_cash = (full_share_count - np.floor(full_share_count)) * new_cost_basics
        self.cost_basis = np.floor(new_cost_basics)
        self.amount = full_share_count
        return left_cash

    def update(self,txn):
        if self.asset != txn.asset:
            raise Exception('transaction asset is different from position asset')
        total_shares = txn.amount + self.amount
        if total_shares == 0:
            # 用于统计transaction是否盈利
            # self.cost_basis = 0.0
            position_return = (txn.price - self.cost_basis)/self.cost_basis
            self.cost_basis = position_return
        elif total_shares < 0:
            raise Exception('for present put action is not allowed')
        else:
            total_cost = txn.amout * txn.price + self.amount * self.cost_basis
            new_cost_basis = total_cost / total_shares
            self.cost_basis = new_cost_basis
        if self.last_sale_dt is None or txn.dt > self.last_sale_dt:
            self.last_sale_dt = txn.dt
            self.last_sale_price = txn.price

        self.amount = total_shares

    def adjust_commission_cost_basis(self,asset,cost):
        """
            成交价格与成本分开
        """
        if asset != self.asset:
            raise Exception('updating a commission for a different asset')
        if cost == 0.0 or self.amount == 0:
            return

        prev_cost = self.amount * self.cost_basis
        new_cost = prev_cost + cost
        self.cost_basis = new_cost / self.amount


    def __repr__(self):
        template = "asset :{asset} , amount:{amount},cost_basis:{cost_basis}"
        return template.format(
            asset = self.asset,
            amount = self.amount,
            cost_basis = self.cost_basis
        )

    def to_dict(self):
        """
            create a dict representing the state of this position
        :return:
        """
        return {
            'sid':self.asset,
            'amount':self.amount,
            'cost_basis':self.cost_basis,
            'last_sale_price':self.last_sale_price
        }

from collections import OrderedDict

class PositionStats(object):

    gross_exposure = None
    gross_value = None
    long_exposure = None
    net_exposure = None
    net_value = None
    short_exposure = None
    longs_count = None
    shorts_count = None
    position_exposure_array = np.array()
    position_exposure_series = pd.series()

    def __new__(cls):
        self = cls()
        es = pd.Series(np.array([],dtype = 'float64'),index = np.array([],dtpe = 'int64'))
        self._underlying_value_array = es.values
        self._underlying_index_array = es.index.values

#涉及仓位的价格同步、统计指标
def update_position_last_sale_prices(positions,get_price,dt):
    for outer_position in positions.values():
        inner_position = outer_position.inner_position
        last_sale_price = get_price(inner_position.asset)
        inner_position.last_sale_price = last_sale_price
        inner_position.last_sale_date = dt


def calculate_position_tracker_stats(positions,stats):
    """
        stats ---- PositionStats
    """
    longs_count = 0
    long_exposure = 0
    shorts_count = 0
    short_exposure = 0

    for outer_position  in positions.values():
        position = outer_position.inner_position
        #daily更新价格
        exposure = position.amount * position.last_sale_price
        if exposure > 0:
            longs_count += 1
            long_exposure += exposure
        elif exposure < 0:
            shorts_count +=1
            short_exposure += exposure
    #
    net_exposure = long_exposure + short_exposure
    gross_exposure = long_exposure - short_exposure

    stats.gross_exposure = gross_exposure
    stats.long_exposure = long_exposure
    stats.longs_count = longs_count
    stats.net_exposure = net_exposure
    stats.short_exposure = short_exposure
    stats.shorts_count = shorts_count


class PositionTracker(object):
    """
        持仓变动
        the current state of position held
    """
    def __init__(self,data_frequency):
        self.positions = OrderedDict()

        #现金分红
        self._unpaid_divdend = defaultdict(list)
        #送股、配股
        self._unpaid_stock_divdends = defaultdict(list)
        self.data_frequency = data_frequency

        #cache the stats until
        self._dirty_stats = True
        self._stats = PositionStats.new()

        # record the algorithm return which decided by asset reason
        self.record_vars = defaultdict(dict)

    # 更新仓位信息 价格 数量 价格 时间
    def update_position(self,
                        asset,
                        amount = None,
                        last_sale_price = None,
                        last_sale_date = None,
                        cost_basis = None):
        self._dirty_stats = True

        if asset not in self.positions:
            position = Position(asset)
        else:
            position = self.position[asset]

        if amount is not None:
            position.amount = amount
        if last_sale_price is not None :
            position.last_sale_price = last_sale_price
        if last_sale_date is not None :
            position.last_sale_date = last_sale_date
        if cost_basis is not None :
            position.cost_basis = cost_basis

    # 执行
    def execute_transaction(self,txn):
        self._dirty_stats = True

        asset = txn.asset

        # 新的股票仓位
        if asset not in self.positions:
            position = Position(asset)
        else:
            position = self.positions[asset]

        position.update(txn)

        if position.amount ==0 :
            #统计策略的对应的收益率
            dt = txn.dt
            algorithm_ret = position.cost_basis
            asset_origin = position.asset.reason
            self.record_vars[asset_origin] = {str(dt):algorithm_ret}

            del self.positions[asset]

    #除权 返回cash
    def handle_spilts(self,splits):
        total_leftover_cash = 0

        for asset,ratio in splits.items():
            if asset in self.positions:
                position = self.positions[asset]
                leftover_cash = position.handle_split(asset,ratio)
                total_leftover_cash += leftover_cash
        return total_leftover_cash

    #将分红或者配股的数据分类存储
    def earn_divdends(self,cash_divdends,stock_divdends):
        """
            given a list of divdends where ex_date all the next_trading
            including divdend and stock_divdend
        """
        for cash_divdend in cash_divdends:
            div_owned = self.positions[cash_divdend['paymen_asset']].earn_divdend(cash_divdend)
            self._unpaid_divdend[cash_divdend.pay_date].apppend(div_owned)

        for stock_divdend in stock_divdends:
            div_owned_ = self.positions[stock_divdend['payment_asset']].earn_stock_divdend(stock_divdend)
            self._unpaid_stock_divdends[stock_divdend.pay_date].append(div_owned_)

    # 根据时间执行分红或者配股
    def pay_divdends(self,next_trading_day):
        """
            股权登记日，股权除息日（为股权登记日下一个交易日）
            但是红股的到账时间不一致（制度是固定的）
            根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
            投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
            其中对于分红的时间存在差异

            根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，

            持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
        """
        net_cash_payment = 0.0

        # cash divdend
        try:
            payments = self._unpaid_divdend[next_trading_day]
            del self._unpaid_divdend[next_trading_day]
        except KeyError:
            payments = []

        for payment in payments:
            net_cash_payment += payment['cash_amount']

        #stock divdend
        try:
            stock_payments = self._unpaid_stock_divdends[next_trading_day]
        except KeyError:
            stock_payments = []

        for stock_payment in stock_payments:
            payment_asset = stock_payment['payment_asset']
            share_amount = stock_payment['share_amount']
            if payment_asset in self.positions:
                position = self.positions[payment_asset]
            else:
                position = self.positions[payment_asset] = Position(payment_asset)
            position.amount  += share_amount
        return net_cash_payment

    def sync_last_sale_prices(self,
                              dt,
                              data_portal):
        """update last_sale_price of position"""
        get_price = partial(data_portal.get_scalar_asset_spot_value,
                            field = 'close',
                            dt = dt,
                            data_frequency = self.data_frequency)

        update_position_last_sale_prices(self.position,get_price,dt)

    @property
    def stats(self):
        """基于sync_last_sale_price  --- 计算每天的暴露度"""
        calculate_position_tracker_stats(self.positions,self._stats)
        return self._stats

    # protocol
    def get_positions(self):
        positions = self._positions_store

        for asset, pos in iteritems(self.positions):
            # Adds the new position if we didn't have one before, or overwrite
            # one we have currently
            positions[asset] = pos.protocol_position

        return positions

#protocol
class MutableView(object):
    """
        A mutable view over an "immutable" object
    """
    __slots__ = ('_mutable_view_obj')

    def __init__(self,ob):
        object.__setattr__(self,'_mutable_view_ob',ob)

    def __getattr__(self, item):
        return getattr(self._mutable_view_ob,item)

    def __setattr(self,attr,value):
        #vars() 函数返回对象object的属性和属性值的字典对象 --- 扩展属性类型
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)'%(type(self).__name__,self._mutable_view_ob)

class Event(object):

    def __init__(self,initial_value = None):
        if initial_value :
            self.__dict__.update(initial_value)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self,other):
        return hasattr(other,'__dict__') and self.___dict__ == other.__dict__

    def __containd__(self,name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self,index = None):
        return pd.tseries(self.__dict__,index = index)


class Portfolio(object):
    """
        基于Portfolio 计算止损点，盈亏点,持仓限制
    """

    def __init__(self,start_date = None,capital_base = 0.0):
        self_ = MutableView(self)
        self_.cash_flow = 0.0
        self_.starting_cash = capital_base
        self_.portfolio_value = capital_base
        self_.pnl = 0.0
        self_.returns = 0.0
        self_.cash = capital_base
        self_.start_date = start_date
        self_.position_value = 0.0
        self_.positions_exposure = 0.0

    @property
    def capital_used(self):
        return self._cash_flow

    def __setattr__(self,name,value):
        raise AttributeError('cannot mutate Portfolio objects')

    def __repr__(self):
        return 'Portofolio({0})'.format(self.__dict__)


class Account(object):
    """
        the account object tracks information about the trading account
    """
    def __init__(self):
        # 避免直接定义属性
        self_ = MutableView(self)
        self_.settle_cash = 0.0
        self_.buying_power = float('inf')
        self_.equity_with_loan = 0.0
        self_.total_position_value = 0.0
        self_.total_position_exposure = 0.0
        self_.cushion = 0.0
        self_.leverage = 0.0

    def __setattr__(self,attr,value):
        raise AttributeError

    def __repr__(self):
        return "Account ({0})".format(self.__dict__)


not_overridden = sentinel(
    'not_over_ridden',
    'mark that an account field has not been overridden'
)


class Ledger(object):
    """
        the ledger tracks all orders and transactions as well as the current state of the portfolio and positions
        逻辑 --- 核心position_tracker （ process_execution ,handle_splits , handle_divdend) --- 生成position_stats
        更新portfolio --- 基于portfolio更新account

    """
    def __init__(self,trading_sessions,capital_base,data_frequency):
        """构建可变、不可变的组合、账户"""
        if not len(trading_sessions):
            raise Exception('calendars must not be null')

        start = trading_sessions[0]

        # here is porfolio
        self._immutable_porfolio = Portfolio(start,capital_base)
        self._portfolio = MutableView(self._immutable_portfolio)
        self._immutable_account = Account()
        self._account = MutableView(self._immutable_account)
        self.position_tracker = PositionTracker(data_frequency)
        self._position_stats = None

        self._processed_transaction = defaultdict(list)
        self._payout_last_sale_price = {}
        self._previous_total_returns = 0
        self._account_overrides = {}
        self.daily_returns_series = pd.Series(np.nan,index = trading_sessions)

    @property
    def todays_returns(self):
        return (
            (self.portfolio.returns +1) /
            (self._previous_total_returns +1 ) - 1
        )

    def start_of_session(self,session_label):
        self._processed_transaction.clear()
        self._prevoius_total_returns = self.portfolio.returns

    def end_of_session(self,session_ix):
        self.daily_returns_series[session_ix] = self.todays_returns

    def sync_last_sale_prices(self,
                              dt,
                              data_portal):
        self.position_tracker.sync_last_sale_prices(
            dt,data_portal,
        )

    @staticmethod
    def _calculate_payout(amount,old_price,price,multiplier = 1):

        return (price - old_price) * multiplier * amount

    def _cash_flow(self,amount):
        """
            update the cash of portfolio
        """
        self._dirty_portfolio = True
        p = self._portfolio
        p.cash_flow += amount
        p.cash += amount

    def process_transaction(self,transaction):
        position = self.position_tracker.positions[asset]
        amount = position.amount
        left_amount = amount + transaction.amount
        if left_amount == 0:
            del self._payout_last_sale_price[asset]
        elif left_amount < 0:
            raise Exception('禁止融券卖出')
        # calculate cash
        self._cash_flow( - transaction.amount * transaction.price)
        #execute transaction
        self.position_tracker.execute_transaction(transaction)
        transaction_dict = transaction.to_dict()
        self._processed_transaction[transaction.dt].append(transaction_dict)

    def process_split(self,splits):
        """
            splits --- (asset,ratio)
        :param splits:
        :return:
        """
        leftover_cash = self.position_tracker.handle_spilts(splits)
        if leftover_cash > 0 :
            self._cash_flow(leftover_cash)

    def process_commission(self,commission):
        asset = commission['asset']
        cost = commission['cost']

        self.position_tracker.handle_commission(asset,cost)
        self._cash_flow(-cost)

    def process_divdends(self,next_session,adjustment_reader):
        """
            基于时间、仓位获取对应的现金分红、股票分红
        """
        position_tracker = self.position_tracker
        #针对字典 --- set return keys
        held_sids = set(position_tracker.positions)
        if held_sids:
            cash_divdend = adjustment_reader.get_dividends_with_ex_date(
                held_sids,
                next_session,
            )
            stock_dividends = (
                adjustment_reader.get_stock_dividends_with_ex_date(
                    held_sids,
                    next_session,
                )
            )
        #添加
        position_tracker.earn_divdends(
            cash_divdend,stock_dividends
        )
        #基于session --- pay_date 处理
        self._cash_flow(
            position_tracker.pay_divdends(next_session)
        )

    # 创建手动关闭订单交易  --- 标的退出
    def manual_close_position(self,asset,dt,data_portal):
        txn = self.position_tracker.maybe_create_close_position_transaction(
            asset,
            dt,
            data_portal
        )
        if txn is not None:
            self.process_transaction(txn)

    #修改或者增加本金
    def capital_change(self,change_amount):
        self.update_portfolio()
        portfolio = self._portfolio

        portfolio.portfolio_value += change_amount
        portfolio.cash += change_amount

    #计算整个组合的收益
    def _get_payout_total(self,positions):
        calculate_payout = self._calculate_payout
        #建仓价格
        payout_last_sale_prices = self._payout_last_sale_price

        total = 0
        for asset ,old_price in payout_last_sale_prices.items():
            position = positions[asset]
            payout_last_sale_prices[asset] = price =  position.last_sale_price
            amount = position.amount
            total += calculate_payout(
                amount,
                old_price,
                price,
                asset.price_multiplier,
            )
        return total

    # 持仓和组合 -- 对应 组合包括cash
    def update_portfolio(self):
        """
            force a computation of the current portfolio
            portofolio 保留最新
        """
        if not self._dirty_portfolio:
            return

        portfolio = self._portfolio
        pt = self.position_tracker

        portfolio.positions = pt.get_positions()
        #计算positioin stats --- sync_last_sale_price
        position_stats = pt.stats

        portfolio.positions_value = position_value = (
            position_stats.net_value
        )

        portfolio.positions_exposure = position_stats.net_exposure
        self._cash_flow(self._get_payout_total(pt.positions))

        # portfolio_value 初始化capital_value
        start_value = portfolio.portfolio_value
        portfolio.portfolio_value = end_value = portfolio.cash + position_value

        # daily 每天账户净值波动
        pnl = end_value - start_value
        if start_value !=0 :
            returns = pnl/start_value
        else:
            returns = 0.0

        #pnl --- 投资收益
        portfolio.pnl += pnl
        # 每天的区间收益率 --- 递归方式
        portfolio.returns = (
            (1+portfolio.returns) *
            (1+returns) - 1
        )
        self._dirty_portfolio = False

    @property
    def portfolio(self):
        self.update_portfolio()
        return self._immutable_porfolio

    #计算杠杆率
    def calculate_period_stats(self):
        position_stats = self.position_tracker.stats
        portfolio_value = self.portfolio.portfolio_value

        if portfolio_value == 0:
            gross_leverage = np.inf
        else:
            gross_leverage = position_stats.gross_exposure / portfolio_value
            net_leverage = position_stats.net_exposure / portfolio_value
        return gross_leverage,net_leverage

    def override_account_fields(self,
                                settled_cash = not_overridden,
                                total_positions_values = not_overridden,
                                total_position_exposure = not_overridden,
                                cushion = not_overridden,
                                gross_leverage = not_overridden,
                                net_leverage = not_overridden,
    ):
        #locals ---函数内部的参数
        self._account_overrides = kwargs = {k:v for k,v in locals().items() if v is not not_overridden}
        del kwargs['self']

    @property
    def account(self):
        portfolio = self.portfolio
        account = self._account

        account.settled_cash = portfolio
        account.total_positions_values = portfolio.portfolio_value - portfolio.cash
        account.total_position_exposure = portfolio.positions_exposure
        account.cushion = portfolio.cash / portfolio.positions_value
        account.gross_leverage,account.net_leverage = self.calculate_period_stats()

        #更新
        for k,v in self._account_overrides:
            setattr(account,k,v)

    #返回 account transaction ,orders, positions
    def transactions(self,dt = None):
        if dt is None:
            #平铺获取
            return [
                txn
                for by_day in self._processed_transaction.values()
                for txn in by_day
            ]
        return self._processed_transaction.get(dt,[])

    @property
    def positions(self):
        # to_dict
        return self.position_tracker.get_position_list()


#controls
from abc import ABC ,abstractmethod

class TradingControl(ABC):
    """
        abstract base class represent a fail-safe control on the behavior of any algorithm
    """
    def __init__(self,on_error,**kwargs):

        self.on_error = on_error
        self._fail_args = kwargs

    @abstractmethod
    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        raise NotImplementedError

    def __repr__(self):
        return "{name}({attr})".format(name = self.__class__.__name__,
                                       attr = self._fail_args)

    def _constraint_msg(self,metadata = None):
        constraint = repr(self)
        if metadata :
            constraint = "{contraint}(Metadata:{metadata})".format(constraint,metadata)
        return constraint

    def handle_violation(self,asset,amount,datetime,metadata = None):
        constraint = self._constraint_msg(metadata)

        if self.on_error == 'fail':
            raise TradingControlViolation(
                asset = asset,
                amount = amount,
                datetime = datetime,
                constraint = constraint
            )
        elif self.on_error == 'log':
            log.error('order for amount shares of asset at dt')


class LongOnly(TradingControl):

    def __init__(self,on_error):
        super(LongOnly,self).__init__(on_error)

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if portfolio.positons[asset].amount + amount  < 0 :
            self.handle_violation(asset,amount,algo_datetime)

class RestrictedListOrder(TradingControl):
    """ represents a restricted list of assets that canont be ordered by the algorithm"""
    def __init__(self,on_error,restrictions):
        super(RestrictedListOrder,self).__init__(on_error)
        self.restrictions = restrictions

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if self.restrictions.is_restricted(asset,algo_datetime):
            self.handle_violation(asset,amount,algo_datetime)

class AssetDateBounds(TradingControl):
    """
        prohibition against an asset before start_date or after end_date
    """
    def __init__(self,on_error):
        super(AssetDateBounds,self).__init__(on_error)

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        asset = txn.asset
        amount = txn.amount
        if amount == 0 :
            return

        normalized_algo_dt = pd.Timestamp(algo_datetime).normalize()

        if asset.start_date :
            normalized_start = pd.Timestamp(asset.start_date).normarlize()
            if normalized_start > normalized_algo_dt :
                metadata = {
                    'asset_start_date':normalized_start
                }
                self.handle_violation(asset,amount,algo_datetime,metadata = metadata)
        if asset.end_date:
            normalized_end = pd.Timestamp(asset.end_date).normalize()
            if normalized_end < normalized_algo_dt :
                metadata = {
                    'asset_end_date':normalized_end
                }
                self.handle_violation(asset,amount,algo_datetime,metadata = metadata)


class MaxPositionOrder(TradingControl):
    """represent a limit on the maximum position size that ca be held by an algo for a given asset
      股票最大持仓比例 asset -- sid reason
    """
    def __init__(self,on_error,
                 max_notional = None):

        self.on_error = on_error
        self.max_notional = max_notional

    def validate(self,
                 txn,
                 portfolio,
                 algo_datetime,
                 algo_current_data):

        amount = txn.amount
        current_price = txn.price

        current_share_count = portfolio.positions[asset].amount
        share_post_order  = current_share_count + amount

        value_post_order_ratio = share_post_order * current_price / portfolio.portfolio_value

        too_many_value =  value_post_order_ratio > self.max_notional

        if too_many_value:
            self.handle_violation(asset,amount,algo_datetime)


#上下文context
class nop_context(object):
    """
        a nop context manager
    """
    def __enter__(self):
        pass

    def __exit__(self):
        pass

def _nop(args,**kwargs):

    pass


class CallbackManager(object):
    """
        create a context manager for a pre-execution callback and a post-execution callback
        context 里面嵌套 context
    """
    def __init__(self,pre = None , post = None):
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self,*args , **kwargs):
        return _MangedCallbackContext(self.pre,self.post,*args,**kwargs)

    def __enter__(self):
        return self.pre

    def __exit__(self,*exec_info):
        self.post()

class _ManagedcallbackContext(object):

    def __init__(self,pre,post,args,kwargs):
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self._pre(*self._args,**self._kwargs)

    def __enter__(self):
        self.post(*self._args,**self._kwargs)


#事件 rule callback  用于scedule module
class EventManager(object):
    """
        manage a list of event objects
        checking the rule and dispatch the handle_data to the events
        event --- rule ,trigger ,function : handle_data
    """
    def __init__(self,create_context = None):
        self._events = []
        # 要不然函数或者类的__call__ 方法
        self._create_context = (
            create_context if create_context is not None
            else  lambda *_ : nop_context
        )

    def add_event(self,event,prepend = False):
        if prepend:
            self._event.insert(0,event)
        else:
            self._events.append(event)

    #与evnet hanle_data方法保持一致
    def  hanle_data(self,context,data,dt):
            with self._create_context(data):
                for event in self._events:
                    event.handle_data(
                        context,
                        data,
                        dt,
                    )

from collections import namedtuple

class Event(namedtuple('Event',['rule','callback'])):
    """
        event consists of rule and callback
        when rule is triggered ,then callback
    """
    def __new__(cls,rule,callback=None):
        callback = callback or (lambda *args,**kwargs : None)
        return super(cls,cls).__new__(cls,rule = rule,callback = callback)

    def handle_data(self,context,data,dt):
        if self.rule.should_trigger(dt):
            self.callback(context,data)

from abc import ABC , abstractmethod

class EventRule(ABC):
    """
        event --- rule
    """
    _cal = None

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self,value):
        self._cal = value

    @abstractmethod
    def should_trigger(self,dt):
        raise NotImplementedError

class StatelessRule(EventRule):
    """
        a stateless rule can be composed to create new rule
    """
    def and_(self,rule):
        """
            trigger only when both rules trigger
        :param rule:
        :return:
        """
        return ComposedRule(self,rule,ComposedRule.lazy_and)

    __and__ = and_

class ComposedRule(StatelessRule):
    """
     compose two rule with some composing function
    """
    def __init__(self,first,second,composer):
        if not (isinstance(first,StatelessRule) and isinstance(second,StatelessRule)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self,dt):

        return self.composer(self.first,self.second)

    @staticmethod
    def lazy_and(first_trigger,second_trigger,dt):
        """
            lazy means : lazy_and will not call whenwhen first_trigger is not Stateless
        :param first_trigger:
        :param second_trigger:
        :param dt:
        :return:
        """
        return first_trigger.should_trigger(dt) and second_trigger.should_trigger(dt)

    @staticmethod
    def cal(self):
        return self.first.cal

    @cal.setter
    def cal(self,value):
        self.first.cal = self.second.cal = value

class Always(StatelessRule):

    @staticmethod
    def always_trigger(dt):
        return True

    should_trigger = always_trigger


class Never(StatelessRule):

    @staticmethod
    def never_trigger():
        return False

    should_trigger = never_trigger

from multiprocessing import Pool

from functools import lru_cache

asset = namedtuple('Asset',['dt','sid','reason','auto_close_date'])

#ziplineA股本地化
from abc import ABC

class Algorithm(ABC):

    def __init__(self,algo_params,data_portal):

        self.algo_params = algo_params
        self.data_portal = data_portal

    def handle_data(self):
        """
            handle_data to run algorithm
        """
    @abstractmethod
    def before_trading_start(self,dt,asset):
        """
            计算持仓股票的卖出信号
        """

    @abstractmethod
    def initialzie(self,dt):
        """
           run algorithm on dt
        """
        pass

class PositionManage(ABC):
    """
        distribution base class
    """
    def __init__(self,data_portal,params = None):
        self.data_portal = data_portal
        self.params = params

    def handle_data(self):
        pass

    @abstractmethod
    def compute_pos_placing(self,assets):
        raise NotImplementedError

class Simple(PositionManage):

    def compute_pos_placing(self,assets):
        asset_weight = { asset: 1 / len(assets) for asset in assets}
        return asset_weight

class Turtle(PositionManage):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例
    """
    def __init__(self,params,data_portal):
        self.turtle_params = params
        self.data_portal = data_portal

    def handle_data(self,sid):
        atr = self.data_portal.get_tr(sid,
                                           self.turtle_params['window'],
                                           self.turtle_params['type'])
        return atr

    @staticmethod
    def _calculate_volatility(self,asset):
        atr = self.handle_data(asset)
        std = atr.std()
        return std

    def compute_pos_placing(self,assets):
        #基于波动确定资金分配 --- 一个波动率点对应资金比例
        turtle_weight = {asset : self._calculate_volatility(asset) for asset in assets}
        return turtle_weight

class Kelly(PositionManage):
    """
        基于策略的胜率反向推导出凯利仓位
    """
    def __init__(self,hitrate):
        self.win_rate = hitrate

    @staticmethod
    def _calculate_kelly(sid):
        rate = self.win_rate[sid.reason]
        return 2 * rate -1

    def compute_pos_placing(self,assets):
        kelly_weight = {asset: self._calculate_kelly(asset) for asset in assets }
        return kelly_weight

class UnionEngine(object):
    """
        组合不同算法---策略
        返回 --- Order对象
    """
    def __init__(self,algo_mappings,data_portal,blotter,assign_policy = Simple):
        self.data_portal = data_portal
        self.postion_allocation = assign_policy
        self.blotter = blotter
        self.loaders = [self.get_loader_class(key,args) for key,args in algo_mappings.items()]

    @staticmethod
    def get_loader_class(key,args):
        """
        :param key: algo_name or algo_path
        :param args: algo_params
        :return: dict -- __name__ : instance
        """

    @lru_cache(maxsize=32)
    def compute_withdraw(self,dt):
        def run(ins):
            result = ins.before_trading_start(dt)
            return result

        with Pool(processes = len(self.loaders)) as pool:
            exit_assets = [pool.apply_async(run,instance)
                            for instance in self.loaders.values]
        return exit_assets

    @lru_cache(maxsize=32)
    def compute_algorithm(self,dt,metrics_tracker):
        unprocessed_loaders = self.tracker_algorithm(metrics_tracker)
        def run(algo):
            ins = self.loaders[algo]
            result = ins.initialize(dt)
            return result

        with Pool(processes=len(self.loaders)) as pool:
            exit_assets = [pool.apply_async(run, algo)
                           for algo in unprocessed_loaders]
        return exit_assets

    def tracker_algorithm(self,metrics_tracker):
        unprocessed_algo = set(self.algorithm_mappings.keys()) - \
                           set(map(lambda x : x.reason ,metrics_tracker.positions.assets))
        return unprocessed_algo

    def position_allocation(self):
        return self.assign_policy.map_allocation(self.tracker_algorithm)

    def _calculate_order_amount(self,asset,dt,total_value):
        """
            calculate how many shares to order based on the position managment
            and price where is assigned to 10% limit in order to carry out order max amount
        """
        preclose = self.data_portal.get_preclose(asset,dt)
        porportion = self.postion_allocation.compute_pos_placing(asset)
        amount = np.floor(porportion * total_value / (preclose * 1.1))
        return amount

    def get_payout(self, dt,metrics_tracker):
        """
        :param metrics_tracker: to get the position
        :return: sell_orders
        """
        assets_of_exit = self.compute_withdraw(dt)
        positions = metrics_tracker.positions
        if assets_of_exit:
            [self.blotter.order(asset,
                                positions[asset].amount)
                                for asset in assets_of_exit]
            cleanup_transactions,additional_commissions = self.blotter.get_transaction(self.data_portal)
            return cleanup_transactions,additional_commissions

    def get_layout(self,dt,metrics_tracker):
        assets = self.compute_algorithm(dt,metrics_tracker)
        avaiable_cash = metrics_tracker.portfolio.cash
        [self.blotter.order(asset,
                            self._calculate_order_amount(asset,dt,avaiable_cash))
                            for asset in assets]
        transactions,new_commissions = self.blotter.get_transaction(self.data_portal)
        return transactions,new_commissions


class SimulationParameters(object):
    def __init__(self,
                 start_session,
                 end_session,
                 trading_calendar,
                 capital_base = DEAFAULT_CAPITAL_BASE,
                 emission_rate = 'daily',
                 data_frequency = 'daily',
                 arena = 'backtest'):

        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None , 'must pass in trading_calendar'
        assert start_session <= end_session , 'period start falls behind end'

__all__ = [
    'EODCancel',
    'FixedSlippage',
    'FixedBasisPointsSlippage',
    'NeverCancel',
    'VolumeShareSlippage',
    'Restriction',
    'StaticRestrictions',
    'HistoricalRestrictions',
    'RESTRICTION_STATES',
    'cancel_policy',
    'commission',
    'date_rules',
    'events',
    'execution',
    'math_utils',
    'slippage',
    'time_rules',
    'calendars',
]

#aasset restriction

import abc
from numpy import vectorize
from functools import partial, reduce
import operator
import pandas as pd
from six import with_metaclass, iteritems
from collections import namedtuple
from toolz import groupby

Restriction = namedtuple(
    'Restriction', ['asset', 'effective_date', 'state']
)


RESTRICTION_STATES = enum(
    'ALLOWED',
    'FROZEN',
)


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


class Restrictions(with_metaclass(abc.ABCMeta)):
    """
    Abstract restricted list interface, representing a set of assets that an
    algorithm is restricted from trading.
    """

    @abc.abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        asset : Asset of iterable of assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or assets restricted on this dt?

        """
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        """Base implementation for combining two restrictions.
        """
        # If the right side is a _UnionRestrictions, defers to the
        # _UnionRestrictions implementation of `|`, which intelligently
        # flattens restricted lists
        if isinstance(other_restriction, _UnionRestrictions):
            return other_restriction | self
        return _UnionRestrictions([self, other_restriction])


class _UnionRestrictions(Restrictions):
    """
    A union of a number of sub restrictions.

    Parameters
    ----------
    sub_restrictions : iterable of Restrictions (but not _UnionRestrictions)
        The Restrictions to be added together

    Notes
    -----
    - Consumers should not construct instances of this class directly, but
      instead use the `|` operator to combine restrictions
    """

    def __new__(cls, sub_restrictions):
        # Filter out NoRestrictions and deal with resulting cases involving
        # one or zero sub_restrictions
        sub_restrictions = [
            r for r in sub_restrictions if not isinstance(r, NoRestrictions)
        ]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]

        new_instance = super(_UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        """
        Overrides the base implementation for combining two restrictions, of
        which the left side is a _UnionRestrictions.
        """
        # Flatten the underlying sub restrictions of _UnionRestrictions
        if isinstance(other_restriction, _UnionRestrictions):
            new_sub_restrictions = \
                self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]

        return _UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return any(
                r.is_restricted(assets, dt) for r in self.sub_restrictions
            )

        return reduce(
            operator.or_,
            (r.is_restricted(assets, dt) for r in self.sub_restrictions)
        )


class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """
    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=False)


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list.
        """
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, self._restricted_set)
        )


class HistoricalRestrictions(Restrictions):
    """
    Historical restrictions stored in memory with effective dates for each
    asset.

    Parameters
    ----------
    restrictions : iterable of namedtuple Restriction
        The restrictions, each defined by an asset, effective date and state
    """

    def __init__(self, restrictions):
        # A dict mapping each asset to its restrictions, which are sorted by
        # ascending order of effective_date
        self._restrictions_by_asset = {
            asset: sorted(
                restrictions_for_asset, key=lambda x: x.effective_date
            )
            for asset, restrictions_for_asset
            in iteritems(groupby(lambda x: x.asset, restrictions))
        }

    def is_restricted(self, assets, dt):
        """
        Returns whether or not an asset or iterable of assets is restricted
        on a dt.
        """
        if isinstance(assets, Asset):
            return self._is_restricted_for_asset(assets, dt)

        is_restricted = partial(self._is_restricted_for_asset, dt=dt)
        return pd.Series(
            index=pd.Index(assets),
            data=vectorize(is_restricted, otypes=[bool])(assets)
        )

    def _is_restricted_for_asset(self, asset, dt):
        state = RESTRICTION_STATES.ALLOWED
        for r in self._restrictions_by_asset.get(asset, ()):
            if r.effective_date > dt:
                break
            state = r.state
        return state == RESTRICTION_STATES.FROZEN


class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list.

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, securities_in_list)
        )



import contextlib

import threading
context = threading.local()

def get_algo_instance():
    return getattr(context, 'algorithm', None)


def set_algo_instance(algo):
    context.algorithm = algo

class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    """

    def __init__(self, algo_instance):
        self.algo_instance = algo_instance

    def __enter__(self):
        """
        Set the given algo instance, storing any previously-existing instance.
        """
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        """
        Restore the algo instance stored in __enter__.
        """
        set_algo_instance(self.old_algo_instance)


from contextlib import ExitStack

class AlgorithmSimulation(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    # def __init__(self,algo,sim_params,data_portal,benchmark_source):
    #
    #     self.algo = algo
    #     self.sim_params = sim_params
    #     self.data_portal = data_portal
    #     self.benchmark = benchmark_source

    def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
                 restrictions, universe_func):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params
        self.data_portal = data_portal
        self.restrictions = restrictions

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo

        # ==============
        # Snapshot Setup
        # ==============

        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data(universe_func)

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None

        self.clock = clock

        self.benchmark_source = benchmark_source

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            if 'algo_dt' not in record.extra:
                record.extra['algo_dt'] = self.simulation_dt

    def get_simulation_dt(self):
        return self.simulation_dt

    #获取交易日数据，封装为一个API(fetch process flush other api)
    def _create_bar_data(self, universe_func):
        return BarData(
            data_portal=self.data_portal,
            simulation_dt_func=self.get_simulation_dt,
            data_frequency=self.sim_params.data_frequency,
            trading_calendar=self.algo.trading_calendar,
            restrictions=self.restrictions,
            universe_func=get_splits_divdend
        )

    def transfrom(self,dt):
        """
        Main generator work loop.
        """
        algo = self.algo
        metrics_tracker = algo.metrics_tracker
        emission_rate = metrics_tracker.emission_rate
        engine = algo.engine
        handle_data = algo.event_manager.handle_data

        metrics_tracker.handle_market_open(dt, algo.data_portal)

        def process_txn_commission(transactions,commissions):
            for txn in transactions:
                metrics_tracker.process_transaction(txn)

            for commission in commissions:
                metrics_tracker.process_commission(commission)

        @contextlib.contextmanager
        def once_a_day(dt):
            payout = engine.get_payout(dt,metrics_tracker)
            try:
                yield payout
            finally:
                layout = engine.get_layout(dt,metrics_tracker)
                process_txn_commission(*layout)

        def on_exit():
            # Remove references to algo, data portal, et al to break cycles
            # and ensure deterministic cleanup of these objects when the
            # simulation finishes.
            self.algo = None
            self.benchmark_source = self.data_portal = None

        with ExitStack() as stack:
            """
            由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
            这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
            enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
            callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
            """
            stack.callback(on_exit())
            stack.enter_context(ZiplineAPI(self.algo))

            for dt in algo.trading_calendar:

                algo.on_dt_changed(dt)
                algo.before_trading_start(self.current_data(dt))
                with once_a_day(dt) as  action:
                    process_txn_commission(*action)
                yield self._get_daily_message(dt, algo, metrics_tracker)

            risk_message = metrics_tracker.handle_simulation_end(
                self.data_portal,
            )
            yield risk_message

    def _get_daily_message(self, dt, algo, metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_market_close(
            dt,
            self.data_portal,
        )
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message


import operator as op

class DailyFieldLedger(object):

    def __init__(self,ledger_field,packet_field = None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.',1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        field = self._packet_field
        packet['daily_perf'][field] = (
            self._get_ledger_field(ledger)
        )


class StartOfPeriodLedgerField(object):
    """Keep track of the value of a ledger field at the start of the period.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger,
                            sessions,
                            benchmark_source):
        self._start_of_simulation = self._get_ledger_field(ledger)

    def start_of_session(self, ledger, session, data_portal):
        self._previous_day = self._get_ledger_field(ledger)

    def _end_of_period(self, sub_field, packet, ledger):
        packet_field = self._packet_field
        packet['cumulative_perf'][packet_field] = self._start_of_simulation
        packet[sub_field][packet_field] = self._previous_day

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       session_ix,
                       data_portal,
                       benchmark_source):
        self._end_of_period('daily_perf', packet, ledger)


class NumTradingDays(object):
    """Report the number of trading days.
    """
    def start_of_simulation(self, *args):
        self._num_trading_days = 0

    def start_of_session(self,*args):
        self._num_trading_days += 1

    def end_of_session(self,
                   packet,
                   ledger,
                   sessions,
                   session_ix,
                   data_portal,
                   benchmark_source):
        packet['cumulative_risk_metrics']['trading_days'] = (
            self._num_trading_days
        )


class PNL(object):
    """Tracks daily and cumulative PNL.
    """
    def start_of_simulation(self,
                            ledger,
                            sessions,
                            benchmark_source):
        self._previous_pnl = 0.0

    def start_of_session(self, ledger,session,data_portal):
        self._previous_pnl = ledger.portfolio.pnl

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        pnl = ledger.portfolio.pnl
        packet['daily']['pnl'] = pnl - self._previous_pnl
        packet['cumulative_perf']['pnl'] = pnl
        self._previous_pnl = pnl


class CashFlow(object):
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """
    def start_of_simulation(self,
                            ledger,
                            sessions,
                            benchmark_source):
        self._previous_cash_flow = 0.0

    def start_of_session(self,ledger,session,data_portal):
        self._previous_cash_flow = ledger.portfolio.cash_flow

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        cash_flow = ledger.portfolio.cash_flow
        packet['daily_perf']['capital_used'] = (
            cash_flow - self._previous_cash_flow
        )
        packet['cumulative_perf']['capital_used'] = cash_flow
        # self._previous_cash_flow = cash_flow


class Transactions(object):
    """Tracks daily transactions.
    """

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        packet['daily_perf']['transactions'] = ledger.transactions()


class Positions(object):
    """Tracks daily positions.
    """

    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        packet['daily_perf']['positions'] = ledger.positions()


class Returns(object):
    """Tracks the daily and cumulative returns of the algorithm.
    """
    def end_of_session(self,
                       packet,
                       ledger,
                       sessions,
                       session_ix,
                       data_portal,
                       benchmark_source):
        packet['daily_perf']['returns'] = ledger.todays_returns
        packet['cumulative_perf']['returns'] = ledger.portfolio.returns


class MaxLeverage(object):
    """Tracks the maximum account leverage.
    """
    def start_of_simulation(self, *args):
        self._max_leverage = 0.0

    def end_of_session(self,
                   packet,
                   ledger,
                   sessions,
                   session_ix,
                   data_portal,
                   benchmark_source):
        self._max_leverage = max(self._max_leverage, ledger.account.leverage)
        packet['cumulative_risk_metrics']['max_leverage'] = self._max_leverage


class ReturnsStatistic(object):
    """A metric that reports an end of simulation scalar or time series
    computed from the algorithm returns.

    Parameters
    ----------
    function : callable
        The function to call on the daily returns.
    field_name : str, optional
        The name of the field. If not provided, it will be
        ``function.__name__``.
    """
    def __init__(self, function, field_name=None):
        if field_name is None:
            field_name = function.__name__

        self._function = function
        self._field_name = field_name

    def end_of_session(self,
                   packet,
                   ledger,
                   sessions,
                   session_ix,
                   data_portal,
                   benchmark_source):
        # res = self._function(ledger.daily_returns_array[:session_ix + 1])
        res = self._function(ledger.daily_returns_array)
        if not np.isfinite(res):
            res = None
        packet['cumulative_risk_metrics'][self._field_name] = res


class BenchmarkReturnsAndVolatility(object):
    """Tracks daily and cumulative returns for the benchmark as well as the
    volatility of the benchmark returns.
    """

    def end_of_session(self,
                          packet,
                          ledger,
                          sessions,
                          session_ix,
                          data_portal,
                          benchmark_source):
        daily_returns_series = benchmark_source.daily_returns(
            sessions[0],
            sessions[session_ix],
        )
        #Series.expanding(self, min_periods=1, center=False, axis=0)
        cumulative_annual_volatility = (
            daily_returns_series.expanding(2).std(ddof=1) * np.sqrt(252)
        ).values[-1]

        cumulative_return = np.cumprod( 1+ daily_returns_series.values) -1

        packet['daily_perf']['benchmark_return'] = daily_returns_series[-1]
        packet['cumulative_perf']['benchmark_return'] = cumulative_return
        packet['cumulative_perf']['benchmark_annual_volatility'] = cumulative_annual_volatility


class AlphaBeta(object):
    """End of simulation alpha and beta to the benchmark.
    """
    def end_of_simulation(self,
                   packet,
                   ledger,
                   sessions,
                   data_portal,
                   benchmark_source):
        risk = packet['cumulative_risk_metrics']
        benchmark_returns =  benchmark_source.daily_returns(
            sessions[0],
            sessions[-1])
        alpha, beta = ep.alpha_beta_aligned(
            ledger.daily_returns_array,
            benchmark_returns)

        if np.isnan(alpha):
            alpha = None
        if np.isnan(beta):
            beta = None

        risk['alpha'] = alpha
        risk['beta'] = beta


class ProbStatistics(object):
    """
        1、度量算法触发的概率（生成transaction)
        2、算法的胜率（产生正的收益概率）--- 当仓位完全退出时
    """
    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions,
                          data_portal,
                          benchmark_source):

        records = ledger.position_tracker.record_vars

        for algo_name , ret_list in records.items():
            packet['cumulative_risk_metrics']['algorithm_ret_list'][algo_name] = ret_list
            hitrate = sum(filter(lambda x: x > 0, ret_list.values())) / len(ret_list)
            packet['cumulative_risk_metrics']['%s_hitrate'%algo_name] = hitrate


class _ClassicRiskMetrics(object):
    """
        Produces original risk packet.
    """

    @classmethod
    def risk_metric_period(cls,
                           start_session,
                           end_session,
                           algorithm_returns,
                           benchmark_returns):
        """
        Creates a dictionary representing the state of the risk report.

        Parameters
        ----------
        start_session : pd.Timestamp
            Start of period (inclusive) to produce metrics on
        end_session : pd.Timestamp
            End of period (inclusive) to produce metrics on
        algorithm_returns : pd.Series(pd.Timestamp -> float)
            Series of algorithm returns as of the end of each session
        benchmark_returns : pd.Series(pd.Timestamp -> float)
            Series of benchmark returns as of the end of each session
        algorithm_leverages : pd.Series(pd.Timestamp -> float)
            Series of algorithm leverages as of the end of each session

        Returns
        -------
        risk_metric : dict[str, any]
            Dict of metrics that with fields like:
                {
                    'algorithm_period_return': 0.0,
                    'benchmark_period_return': 0.0,
                    'treasury_period_return': 0,
                    'excess_return': 0.0,
                    'alpha': 0.0,
                    'beta': 0.0,
                    'sharpe': 0.0,
                    'sortino': 0.0,
                    'period_label': '1970-01',
                    'trading_days': 0,
                    'algo_volatility': 0.0,
                    'benchmark_volatility': 0.0,
                    'max_drawdown': 0.0,
                    'max_leverage': 0.0,
                }
        """

        algorithm_returns = algorithm_returns[
            (algorithm_returns.index >= start_session) &
            (algorithm_returns.index <= end_session)
        ]

        # Benchmark needs to be masked to the same dates as the algo returns
        benchmark_returns = benchmark_returns[
            (benchmark_returns.index >= start_session) &
            (benchmark_returns.index <= algorithm_returns.index[-1])
        ]

        benchmark_period_returns = ep.cum_returns(benchmark_returns).iloc[-1]
        algorithm_period_returns = ep.cum_returns(algorithm_returns).iloc[-1]

        #组合胜率、超额胜率、
        overall_hitrate = [algorithm_period_returns > 0].sum() / len(algorithm_period_returns)
        excess_hitrate = (algorithm_period_returns > benchmark_period_returns).sum() / len(algorithm_period_returns)

        alpha, beta = ep.alpha_beta_aligned(
            algorithm_returns.values,
            benchmark_returns.values,
        )

        sharpe = ep.sharpe_ratio(algorithm_returns)

        # The consumer currently expects a 0.0 value for sharpe in period,
        # this differs from cumulative which was np.nan.
        # When factoring out the sharpe_ratio, the different return types
        # were collapsed into `np.nan`.
        # TODO: Either fix consumer to accept `np.nan` or make the
        # `sharpe_ratio` return type configurable.
        # In the meantime, convert nan values to 0.0
        if pd.isnull(sharpe):
            sharpe = 0.0

        sortino = ep.sortino_ratio(
            algorithm_returns.values,
            # 回撤
            _downside_risk=ep.downside_risk(algorithm_returns.values),
        )

        rval = {
            'algorithm_period_return': c,
            'benchmark_period_return': benchmark_period_returns,
            'treasury_period_return': 0,
            'excess_return': algorithm_period_returns,
            'alpha': alpha,
            'beta': beta,
            'sharpe': sharpe,
            'sortino': sortino,
            'period_label': end_session.strftime("%Y-%m"),
            'trading_days': len(benchmark_returns),
            'algo_volatility': ep.annual_volatility(algorithm_returns),
            'benchmark_volatility': ep.annual_volatility(benchmark_returns),
            'max_drawdown': ep.max_drawdown(algorithm_returns.values),
            'winrate':overall_hitrate,
            'excess_winrate':excess_hitrate,
        }

        # check if a field in rval is nan or inf, and replace it with None
        # except period_label which is always a str
        return {
            k: (
                None
                if k != 'period_label' and not np.isfinite(v) else
                v
            )
            for k, v in iteritems(rval)
        }

    @classmethod
    def _periods_in_range(cls,
                          months,
                          end_session,
                          end_date,
                          algorithm_returns,
                          benchmark_returns,
                          algorithm_leverages,
                          months_per):
        if months.size < months_per:
            return

        end_date = end_date.tz_convert(None)
        for period_timestamp in months:
            period = period_timestamp.to_period(freq='%dM' % months_per)
            if period.end_time > end_date:
                break

            yield cls.risk_metric_period(
                start_session=period.start_time,
                end_session=min(period.end_time, end_session),
                algorithm_returns=algorithm_returns,
                benchmark_returns=benchmark_returns,
                algorithm_leverages=algorithm_leverages,
            )

    @classmethod
    def risk_report(cls,
                    algorithm_returns,
                    benchmark_returns,
                    algorithm_leverages):
        start_session = algorithm_returns.index[0]
        end_session = algorithm_returns.index[-1]
        # 下个月第一天
        end = end_session.replace(day=1) + relativedelta(months=1)
        months = pd.date_range(
            start=start_session,
            # Ensure we have at least one month
            end=end - datetime.timedelta(days=1),
            freq='M',
            tz='utc',
        )

        periods_in_range = partial(
            cls._periods_in_range,
            months=months,
            end_session=end_session.tz_convert(None),
            end_date=end,
            algorithm_returns=algorithm_returns,
            benchmark_returns=benchmark_returns,
            algorithm_leverages=algorithm_leverages,
        )

        return {
            'one_month': list(periods_in_range(months_per=1)),
            'three_month': list(periods_in_range(months_per=3)),
            'six_month': list(periods_in_range(months_per=6)),
            'twelve_month': list(periods_in_range(months_per=12)),
        }

    def end_of_simulation(self,
                          packet,
                          ledger,
                          sessions,
                          data_portal,
                          benchmark_source):
        packet.update(self.risk_report(
            algorithm_returns=ledger.daily_returns_series,
            benchmark_returns=benchmark_source.daily_returns(
                sessions[0],
                sessions[-1],
            ),
        ))


class MetricsTracker(object):
    """The algorithm's interface to the registered risk and performance
    metrics.

    Parameters
    ----------
    trading_calendar : TrandingCalendar
        The trading calendar used in the simulation.
    first_session : pd.Timestamp
        The label of the first trading session in the simulation.
    last_session : pd.Timestamp
        The label of the last trading session in the simulation.
    capital_base : float
        The starting capital for the simulation.
    emission_rate : {'daily', 'minute'}
        How frequently should a performance packet be generated?
    data_frequency : {'daily', 'minute'}
        The data frequency of the data portal.
    asset_finder : AssetFinder
        The asset finder used in the simulation.
    metrics : list[Metric]
        The metrics to track.
    """
    _hooks = (
        'start_of_simulation',
        'end_of_simulation',

        'start_of_session',
        'end_of_session',
    )

    def __init__(self,
                 trading_calendar,
                 first_session,
                 last_session,
                 capital_base,
                 emission_rate,
                 data_frequency,
                 metrics):
        self.emission_rate = emission_rate

        self._trading_calendar = trading_calendar
        self._first_session = first_session
        self._last_session = last_session
        self._capital_base = capital_base

        self._current_session = first_session

        self._session_count = 0

        self._sessions = sessions = trading_calendar.sessions_in_range(
            first_session,
            last_session,
        )
        self._total_session_count = len(sessions)

        self._ledger = Ledger(sessions, capital_base, data_frequency)

        # bind all of the hooks from the passed metric objects.
        for hook in self._hooks:
            registered = []
            for metric in metrics:
                try:
                    registered.append(getattr(metric, hook))
                except AttributeError:
                    pass

            def closing_over_loop_variables_is_hard(registered=registered):
                def hook_implementation(*args, **kwargs):
                    for impl in registered:
                        impl(*args, **kwargs)

                return hook_implementation
            #属性 --- 方法
            hook_implementation = closing_over_loop_variables_is_hard()

            hook_implementation.__name__ = hook
            # 属性 --- 方法
            setattr(self, hook, hook_implementation)

    def handle_start_of_simulation(self, benchmark_source):
        self._benchmark_source = benchmark_source

        self.start_of_simulation(
            self._ledger,
            self._sessions,
            benchmark_source,
        )

    @property
    def portfolio(self):
        return self._ledger.portfolio

    @property
    def account(self):
        return self._ledger.account

    @property
    def positions(self):
        return self._ledger.position_tracker.positions

    def update_position(self,
                        asset,
                        amount=None,
                        last_sale_price=None,
                        last_sale_date=None,
                        cost_basis=None):
        self._ledger.position_tracker.update_position(
            asset,
            amount,
            last_sale_price,
            last_sale_date,
            cost_basis,
        )

    def override_account_fields(self, **kwargs):
        self._ledger.override_account_fields(**kwargs)

    def process_transaction(self, transaction):
        self._ledger.process_transaction(transaction)

    def handle_splits(self, splits):
        self._ledger.process_splits(splits)

    def process_order(self, event):
        self._ledger.process_order(event)

    def process_commission(self, commission):
        self._ledger.process_commission(commission)

    def process_close_position(self, asset, dt, data_portal):
        self._ledger.close_position(asset, dt, data_portal)

    def capital_change(self, amount):
        self._ledger.capital_change(amount)

    def sync_last_sale_prices(self,
                              dt,
                              data_portal,
                              handle_non_market_minutes=False):
        self._ledger.sync_last_sale_prices(
            dt,
            data_portal,
            handle_non_market_minutes=handle_non_market_minutes,
        )


    def handle_market_open(self, session_label, data_portal):
        """Handles the start of each session.

        Parameters
        ----------
        session_label : Timestamp
            The label of the session that is about to begin.
        data_portal : DataPortal
            The current data portal.
        """
        ledger = self._ledger
        # 账户初始化
        ledger.start_of_session(session_label)

        adjustment_reader = data_portal.adjustment_reader
        if adjustment_reader is not None:
            # this is None when running with a dataframe source
            ledger.process_dividends(
                session_label,
                self._asset_finder,
                adjustment_reader,
            )

        # self._current_session = session_label
        #
        # cal = self._trading_calendar
        # self._market_open, self._market_close = self._execution_open_and_close(
        #     cal,
        #     session_label,
        # )

        self.start_of_session(ledger, session_label, data_portal)

    def handle_market_close(self, dt, data_portal):
        """Handles the close of the given day.

        Parameters
        ----------
        dt : Timestamp
            The most recently completed simulation datetime.
        data_portal : DataPortal
            The current data portal.

        Returns
        -------
        A daily perf packet.
        """
        completed_session = self._current_session

        if self.emission_rate == 'daily':
            # this method is called for both minutely and daily emissions, but
            # this chunk of code here only applies for daily emissions. (since
            # it's done every minute, elsewhere, for minutely emission).
            self.sync_last_sale_prices(dt, data_portal)

        session_ix = self._session_count
        # increment the day counter before we move markers forward.
        self._session_count += 1

        packet = {
            'period_start': self._first_session,
            'period_end': self._last_session,
            'capital_base': self._capital_base,
            'daily_perf': {},
            'cumulative_perf': {},
            'cumulative_risk_metrics': {},
        }
        ledger = self._ledger
        ledger.end_of_session(session_ix)
        self.end_of_session(
            packet,
            ledger,
            completed_session,
            session_ix,
            data_portal,
            self._benchmark_source
        )

        return packet

    def handle_simulation_end(self, data_portal):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """
        import logging
        logging.info(
            'Simulated {} trading days\n'
            'first open: {}\n'
            'last close: {}',
            self._session_count,
            self._trading_calendar.session_open(self._first_session),
            self._trading_calendar.session_close(self._last_session),
        )

        packet = {}
        self.end_of_simulation(
            packet,
            self._ledger,
            self._sessions,
            data_portal,
            self._benchmark_source,
        )
        return packet


class TradingAlgorithm(object):
    """
        position_class : short position and long position
        Position  feature : allocation --- based on the volatity of assets or fixed allocation alogrithm'
        sell_pos : fixed amount ,e.g.:100% or 2/3 or other
        buy_pos :  based on the total number of buy_orders

        buy position : default --- average | manual | vati
        key is to allocate on the all buy_orders at meanwhile as a whole

        借鉴海龟交易算法，以近期的波动率为基础计算仓位
        研究如何把kelly 公式思想融入实际应用中
        无差别分配，避免由于仓位的倾斜导致过度优化
        sell position : fixed  ,distinct to  buy position where can sololy process the sell action
        Sliappage : adjust the price that buy or sell in order to simulate the reality
        prefix intend to specify the order which strategy assign
        (每一个策略得出的标的，生产的订单附有该特征的属性)
        Account : property --- impl_dt asset balance cash total ,limit --- T+0 | T+1 determine the sell action;
        --- the num of holding asset; --- the proportion of each assets;
        非融资融券，持仓个数对应策略个数，每个策略对应相应的持仓标的，所有策略必须每天都跑，持仓每个交易日不一定都是满足最大持仓
        个数，因为不是所有策略都能产生信号。
        基于原则，账户资金充分利用，不一定强制每天持仓都达到上限，只要信号产生并余额允许就执行买入算法
        T +1 --- 当天构建的订单不能处理,针对这个处理方式 --- 设置 订单处理顺序 ，sell --- buy 避开了 T + 1 限制
        需要改进的地方： 目前仅支持买入或者卖出股票必须当天完成，不能分步建仓或者清仓
        单子开启了特征收集，将收集的特征添加到对应的交易中

        A class that represents a trading strategy and parameters to execute
        the strategy.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to ``initialize`` unless listed below.
        initialize : callable[context -> None], optional
            Function that is called at the start of the simulation to
            setup the initial context.
        handle_data : callable[(context, data) -> None], optional
            Function called on every bar. This is where most logic should be
            implemented.
        before_trading_start : callable[(context, data) -> None], optional
            Function that is called before any bars have been processed each
            day.
        analyze : callable[(context, DataFrame) -> None], optional
            Function that is called at the end of the backtest. This is passed
            the context and the performance results for the backtest.
        script : str, optional
            Algoscript that contains the definitions for the four algorithm
            lifecycle functions and any supporting code.
        namespace : dict, optional
            The namespace to execute the algoscript in. By default this is an
            empty namespace that will include only python built ins.
        algo_filename : str, optional
            The filename for the algoscript. This will be used in exception
            tracebacks. default: '<string>'.
        data_frequency : {'daily', 'minute'}, optional
            The duration of the bars.
        equities_metadata : dict or DataFrame or file-like object, optional
            If dict is provided, it must have the following structure:
            * keys are the identifiers
            * values are dicts containing the metadata, with the metadata
              field name as the key
            If pandas.DataFrame is provided, it must have the
            following structure:
            * column names must be the metadata fields
            * index must be the different asset identifiers
            * array contents should be the metadata value
            If an object with a ``read`` method is provided, ``read`` must
            return rows containing at least one of 'sid' or 'symbol' along
            with the other metadata fields.
        identifiers : list, optional
            Any asset identifiers that are not provided in the
            equities_metadata, but will be traded by this TradingAlgorithm.
        get_pipeline_loader : callable[BoundColumn -> pipeline], optional
            The function that maps Pipeline columns to their loaders.
        create_event_context : callable[BarData -> context manager], optional
            A function used to create a context mananger that wraps the
            execution of all events that are scheduled for a bar.
            This function will be passed the data for the bar and should
            return the actual context manager that will be entered.
        history_container_class : type, optional
            The type of history container to use. default: HistoryContainer
        platform : str, optional
            The platform the simulation is running on. This can be queried for
            in the simulation with ``get_environment``. This allows algorithms
            to conditionally execute code based on platform it is running on.
            default: 'zipline'
        adjustment_reader : AdjustmentReader
            The interface to the adjustments.
    """
    def __init__(self,
                 sim_params,
                 data_portal=None,
                 asset_finder=None,
                 # algorithm API
                 namespace=None,
                 script=None,
                 algo_filename=None,
                 initialize=None,
                 handle_data=None,
                 before_trading_start=None,
                 analyze=None,
                 #
                 trading_calendar=None,
                 metrics_set=None,
                 blotter=None,
                 blotter_class=None,
                 cancel_policy=None,
                 benchmark_sid=None,
                 benchmark_returns=None,
                 platform='zipline',
                 capital_changes=None,
                 get_pipeline_loader=None,
                 create_event_context=None,
                 **initialize_kwargs):

        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        self._recorded_vars = {}
        self.namespace = namespace or {}

        self._platform = platform
        self.logger = None

        self.data_portal = data_portal

        if self.data_portal is None:
            if asset_finder is None:
                raise ValueError(
                    "Must pass either data_portal or asset_finder "
                    "to TradingAlgorithm()"
                )
            self.asset_finder = asset_finder
        else:
            # Raise an error if we were passed two different asset finders.
            # There's no world where that's a good idea.
            if asset_finder is not None \
               and asset_finder is not data_portal.asset_finder:
                raise ValueError(
                    "Inconsistent asset_finders in TradingAlgorithm()"
                )
            self.asset_finder = data_portal.asset_finder

        self.benchmark_returns = benchmark_returns

        self.sim_params = sim_params
        if trading_calendar is None:
            self.trading_calendar = sim_params.trading_calendar
        elif trading_calendar.name == sim_params.trading_calendar.name:
            self.trading_calendar = sim_params.trading_calendar
        else:
            raise ValueError(
                "Conflicting trading-calendars: trading_calendar={}, but "
                "sim_params.trading_calendar={}".format(
                    trading_calendar.name,
                    self.sim_params.trading_calendar.name,
                )
            )

        self._last_sync_time = pd.NaT
        self.metrics_tracker = None
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set('default')

        if blotter is not None:
            self.blotter = blotter
        else:
            cancel_policy = cancel_policy or NeverCancel()
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(cancel_policy=cancel_policy)

        # The symbol lookup date specifies the date to use when resolving
        # symbols to sids, and can be set using set_symbol_lookup_date()
        self._symbol_lookup_date = None

        self.event_manager = EventManager(create_event_context)

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = script

        self._handle_data = None

        if self.algoscript is not None:
            unexpected_api_methods = set()
            if initialize is not None:
                unexpected_api_methods.add('initialize')
            if handle_data is not None:
                unexpected_api_methods.add('handle_data')
            if before_trading_start is not None:
                unexpected_api_methods.add('before_trading_start')
            if analyze is not None:
                unexpected_api_methods.add('analyze')

            if unexpected_api_methods:
                raise ValueError(
                    "TradingAlgorithm received a script and the following API"
                    " methods as functions:\n{funcs}".format(
                        funcs=unexpected_api_methods,
                    )
                )

            if algo_filename is None:
                algo_filename = '<string>'
            #exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
            code = compile(self.algoscript, algo_filename, 'exec')
            #动态执行文件， 相当于import
            exec_(code, self.namespace)

            def noop(*args, **kwargs):
                pass

            #dict get参数可以为方法或者默认参数
            self._initialize = self.namespace.get('initialize', noop)
            self._handle_data = self.namespace.get('handle_data', noop)
            self._before_trading_start = self.namespace.get(
                'before_trading_start',
            )
            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get('analyze')

        else:
            self._initialize = initialize or (lambda self: None)
            self._handle_data = handle_data
            self._before_trading_start = before_trading_start
            self._analyze = analyze

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )

        if self.sim_params.capital_base <= 0:
            raise ZeroCapitalError()

        # Prepare the algo for initialization
        self.initialized = False

        self.initialize_kwargs = initialize_kwargs or {}

        self.benchmark_sid = benchmark_sid

        # A dictionary of capital changes, keyed by timestamp, indicating the
        # target/delta of the capital changes, along with values
        self.capital_changes = capital_changes or {}

        # A dictionary of the actual capital change deltas, keyed by timestamp
        self.capital_change_deltas = {}

        self.restrictions = NoRestrictions()

        self._backwards_compat_universe = None

        # Initialize pipe API data.
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}

        # Create an already-expired cache so that we compute the first time
        # data is requested.
        self._pipeline_cache = ExpiringCache(
            cleanup=clear_dataframe_indexer_caches
        )

        self._initialize = None
        self._before_trading_start = None
        self._analyze = None

        self._in_before_trading_start = False


    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

    def init_engine(self, get_loader):
        """
        Construct and store a PipelineEngine from loader.

        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        if get_loader is not None:
            self.engine = SimplePipelineEngine(
                get_loader,
                self.asset_finder,
                self.default_pipeline_domain(self.trading_calendar),
            )
        else:
            self.engine = ExplodingPipelineEngine()


    def compute_eager_pipelines(self):
        """
        Compute any pipelines attached with eager=True.
        """
        for name, pipe in self._pipelines.items():
            if pipe.eager:
                self.pipeline_output(name)

    def before_trading_start(self,dt):

        self.compute_eager_pipelines()


        assets_we_care = self.metrics_tracker.position.assets
        splits = self.data_portal.get_splits(assets_we_care, dt)
        self.metrics_tracker.process_splits(splits)

        self.compute_eager_pipelines()

        self._in_before_trading_start = True

    def handle_data(self, data):
        if self._handle_data:
            self._handle_data(self, data)

    def _create_benchmark_source(self):
        if self.benchmark_sid is not None:
            benchmark_asset = self.asset_finder.retrieve_asset(
                self.benchmark_sid
            )
            benchmark_returns = None
        else:
            if self.benchmark_returns is None:
                raise ValueError("Must specify either benchmark_sid "
                                 "or benchmark_returns.")
            benchmark_asset = None
            # get benchmark info from trading environment, which defaults to
            # downloading data from IEX Trading.
            benchmark_returns = self.benchmark_returns
        return BenchmarkSource(
            benchmark_asset=benchmark_asset,
            benchmark_returns=benchmark_returns,
            trading_calendar=self.trading_calendar,
            sessions=self.sim_params.sessions,
            data_portal=self.data_portal,
            emission_rate=self.sim_params.emission_rate,
        )

    def _create_metrics_tracker(self):
        #'start_of_simulation','end_of_simulation','start_of_session'，'end_of_session','end_of_bar'
        return MetricsTracker(
            trading_calendar=self.trading_calendar,
            first_session=self.sim_params.start_session,
            last_session=self.sim_params.end_session,
            capital_base=self.sim_params.capital_base,
            emission_rate=self.sim_params.emission_rate,
            data_frequency=self.sim_params.data_frequency,
            asset_finder=self.asset_finder,
            metrics=self._metrics_set,
        )

    def _create_generator(self, sim_params):
        if sim_params is not None:
            self.sim_params = sim_params

        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()

        # Set the dt initially to the period start by forcing it to change.
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True

        benchmark_source = self._create_benchmark_source()

        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            benchmark_source,
            self.restrictions,
            universe_func=self._calculate_universe
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    def analyze(self, perf):
        # 分析stats
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def run(self, data_portal=None):
        """Run the algorithm.
        """
        # HACK: I don't think we really want to support passing a data portal
        # this late in the long term, but this is needed for now for backwards
        # compat downstream.
        if data_portal is not None:
            self.data_portal = data_portal
            self.asset_finder = data_portal.asset_finder
        elif self.data_portal is None:
            raise RuntimeError(
                "No data portal in TradingAlgorithm.run().\n"
                "Either pass a DataPortal to TradingAlgorithm() or to run()."
            )
        else:
            assert self.asset_finder is not None, \
                "Have data portal without asset_finder."

        # Create zipline and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)

            # convert perf dict to pandas dataframe
            daily_stats = self._create_daily_stats(perfs)

            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        # TODO: the loop here could overwrite expected properties
        # of daily_perf. Could potentially raise or log a
        # warning.
        for perf in perfs:
            if 'daily_perf' in perf:

                perf['daily_perf'].update(
                    perf['daily_perf'].pop('recorded_vars')
                )
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf

        daily_dts = pd.DatetimeIndex(
            [p['period_close'] for p in daily_perfs], tz='UTC'
        )
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    #根据dt获取change,动态计算，更新数据
    def calculate_capital_changes(self, dt, emission_rate, is_interday,
                                  portfolio_value_adjustment=0.0):
        """
        If there is a capital change for a given dt, this means the the change
        occurs before `handle_data` on the given dt. In the case of the
        change being a target value, the change will be computed on the
        portfolio value according to prices at the given dt

        `portfolio_value_adjustment`, if specified, will be removed from the
        portfolio_value of the cumulative performance when calculating deltas
        from target capital changes.
        """
        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return

        self._sync_last_sale_prices()
        if capital_change['type'] == 'target':
            target = capital_change['value']
            capital_change_amount = (
                target -
                (
                    self.portfolio.portfolio_value -
                    portfolio_value_adjustment
                )
            )

            log.info('Processing capital change to target %s at %s. Capital '
                     'change delta is %s' % (target, dt,
                                             capital_change_amount))
        elif capital_change['type'] == 'delta':
            target = None
            capital_change_amount = capital_change['value']
            log.info('Processing capital change of delta %s at %s'
                     % (capital_change_amount, dt))
        else:
            log.error("Capital change %s does not indicate a valid type "
                      "('target' or 'delta')" % capital_change)
            return

        self.capital_change_deltas.update({dt: capital_change_amount})
        self.metrics_tracker.capital_change(capital_change_amount)

        yield {
            'capital_change':
                {'date': dt,
                 'type': 'cash',
                 'target': target,
                 'delta': capital_change_amount}
        }

    @api_method
    def get_environment(self, field='platform'):
        """Query the execution environment.

        Parameters
        ----------
        field : {'platform', 'arena', 'data_frequency',
                 'start', 'end', 'capital_base', 'platform', '*'}
            The field to query. The options have the following meanings:
              arena : str
                  The arena from the simulation parameters. This will normally
                  be ``'backtest'`` but some systems may use this distinguish
                  live trading from backtesting.
              data_frequency : {'daily', 'minute'}
                  data_frequency tells the algorithm if it is running with
                  daily data or minute data.
              start : datetime
                  The start date for the simulation.
              end : datetime
                  The end date for the simulation.
              capital_base : float
                  The starting capital for the simulation.
              platform : str
                  The platform that the code is running on. By default this
                  will be the string 'zipline'. This can allow algorithms to
                  know if they are running on the Quantopian platform instead.
              * : dict[str -> any]
                  Returns all of the fields in a dictionary.

        Returns
        -------
        val : any
            The value for the field queried. See above for more information.

        Raises
        ------
        ValueError
            Raised when ``field`` is not a valid option.
        """
        env = {
            'arena': self.sim_params.arena,
            'data_frequency': self.sim_params.data_frequency,
            'start': self.sim_params.first_open,
            'end': self.sim_params.last_close,
            'capital_base': self.sim_params.capital_base,
            'platform': self._platform
        }
        if field == '*':
            return env
        else:
            try:
                return env[field]
            except KeyError:
                raise ValueError(
                    '%r is not a valid field for get_environment' % field,
                )

    def add_event(self, rule, callback):
        """Adds an event to the algorithm's EventManager.

        Parameters
        ----------
        rule : EventRule
            The rule for when the callback should be triggered.
        callback : callable[(context, data) -> None]
            The function to execute when the rule is triggered.
        """
        self.event_manager.add_event(
            zipline.utils.events.Event(rule, callback),
        )

    @api_method
    def schedule_function(self,
                          func,
                          date_rule=None,
                          time_rule=None,
                          half_days=True,
                          calendar=None):
        """
        Schedule a function to be called repeatedly in the future.

        Parameters
        ----------
        func : callable
            The function to execute when the rule is triggered. ``func`` should
            have the same signature as ``handle_data``.
        date_rule : zipline.utils.events.EventRule, optional
            Rule for the dates on which to execute ``func``. If not
            passed, the function will run every trading day.
        time_rule : zipline.utils.events.EventRule, optional
            Rule for the time at which to execute ``func``. If not passed, the
            function will execute at the end of the first market minute of the
            day.
        half_days : bool, optional
            Should this rule fire on half days? Default is True.
        calendar : Sentinel, optional
            Calendar used to compute rules that depend on the trading calendar.

        See Also
        --------
        :class:`zipline.api.date_rules`
        :class:`zipline.api.time_rules`
        """

        # When the user calls schedule_function(func, <time_rule>), assume that
        # the user meant to specify a time rule but no date rule, instead of
        # a date rule and no time rule as the signature suggests
        if isinstance(date_rule, (AfterOpen, BeforeClose)) and not time_rule:
            warnings.warn('Got a time rule for the second positional argument '
                          'date_rule. You should use keyword argument '
                          'time_rule= when calling schedule_function without '
                          'specifying a date_rule', stacklevel=3)

        date_rule = date_rule or date_rules.every_day()
        time_rule = ((time_rule or time_rules.every_minute())
                     if self.sim_params.data_frequency == 'minute' else
                     # If we are in daily mode the time_rule is ignored.
                     time_rules.every_minute())

        # Check the type of the algorithm's schedule before pulling calendar
        # Note that the ExchangeTradingSchedule is currently the only
        # TradingSchedule class, so this is unlikely to be hit
        if calendar is None:
            cal = self.trading_calendar
        elif calendar is calendars.US_EQUITIES:
            cal = get_calendar('XNYS')
        elif calendar is calendars.US_FUTURES:
            cal = get_calendar('us_futures')
        else:
            raise ScheduleFunctionInvalidCalendar(
                given_calendar=calendar,
                allowed_calendars=(
                    '[trading-calendars.US_EQUITIES, trading-calendars.US_FUTURES]'
                ),
            )

        self.add_event(
            make_eventrule(date_rule, time_rule, cal, half_days),
            func,
        )

    def make_eventrule(date_rule, time_rule, cal, half_days=True):
        """
        Constructs an event rule from the factory api.
        """
        _check_if_not_called(date_rule)
        _check_if_not_called(time_rule)

        if half_days:
            inner_rule = date_rule & time_rule
        else:
            inner_rule = date_rule & time_rule & NotHalfDay()

        opd = OncePerDay(rule=inner_rule)
        # This is where a scheduled function's rule is associated with a calendar.
        opd.cal = cal
        return opd

    @api_method
    def record(self, *args, **kwargs):
        """Track and record values each day.

        Parameters
        ----------
        **kwargs
            The names and values to record.

        Notes
        -----
        These values will appear in the performance packets and the performance
        dataframe passed to ``analyze`` and returned from
        :func:`~zipline.run_algorithm`.
        """
        # Make 2 objects both referencing the same iterator
        args = [iter(args)] * 2

        # Zip generates list entries by calling `next` on each iterator it
        # receives.  In this case the two iterators are the same object, so the
        # call to next on args[0] will also advance args[1], resulting in zip
        # returning (a,b) (c,d) (e,f) rather than (a,a) (b,b) (c,c) etc.
        positionals = zip(*args)
        for name, value in chain(positionals, iteritems(kwargs)):
            self._recorded_vars[name] = value

    @api_method
    def set_benchmark(self, benchmark):
        """Set the benchmark asset.

        Parameters
        ----------
        benchmark : zipline.assets.Asset
            The asset to set as the new benchmark.

        Notes
        -----
        Any dividends payed out for that new benchmark asset will be
        automatically reinvested.
        """
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()

        self.benchmark_sid = benchmark

    @api_method
    @preprocess(
        symbol_str=ensure_upper_case,
        country_code=optionally(ensure_upper_case),
    )
    def symbol(self, symbol_str, country_code=None):
        """Lookup an Equity by its ticker symbol.

        Parameters
        ----------
        symbol_str : str
            The ticker symbol for the equity to lookup.
        country_code : str or None, optional
            A country to limit symbol searches to.

        Returns
        -------
        equity : zipline.assets.Equity
            The equity that held the ticker symbol on the current
            symbol lookup date.

        Raises
        ------
        SymbolNotFound
            Raised when the symbols was not held on the current lookup date.

        See Also
        --------
        :func:`zipline.api.set_symbol_lookup_date`
        """
        # If the user has not set the symbol lookup date,
        # use the end_session as the date for symbol->sid resolution.
        # self.asset_finder.retrieve_asset(sid)
        _lookup_date = self._symbol_lookup_date \
            if self._symbol_lookup_date is not None \
            else self.sim_params.end_session

        return self.asset_finder.lookup_symbol(
            symbol_str,
            as_of_date=_lookup_date,
            country_code=country_code,
        )

    @property
    def recorded_vars(self):
        return copy(self._recorded_vars)

    def _sync_last_sale_prices(self, dt=None):
        """Sync the last sale prices on the metrics tracker to a given
        datetime.

        Parameters
        ----------
        dt : datetime
            The time to sync the prices to.

        Notes
        -----
        This call is cached by the datetime. Repeated calls in the same bar
        are cheap.
        """
        if dt is None:
            dt = self.datetime

        if dt != self._last_sync_time:
            self.metrics_tracker.sync_last_sale_prices(
                dt,
                self.data_portal,
            )
            self._last_sync_time = dt

    @property
    def portfolio(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.portfolio

    @property
    def account(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.account

    def set_logger(self, logger):
        self.logger = logger

    def on_dt_changed(self, dt):
        """
        Callback triggered by the simulation loop whenever the current dt
        changes.

        Any logic that should happen exactly once at the start of each datetime
        group should happen here.
        """
        self.datetime = dt
        self.blotter.set_date(dt)

    @api_method
    @preprocess(tz=coerce_string(pytz.timezone))
    @expect_types(tz=optional(tzinfo))
    def get_datetime(self, tz=None):
        """
        Returns the current simulation datetime.

        Parameters
        ----------
        tz : tzinfo or str, optional
            The timezone to return the datetime in. This defaults to utc.

        Returns
        -------
        dt : datetime
            The current simulation datetime converted to ``tz``.
        """
        dt = self.datetime
        assert dt.tzinfo == pytz.utc, "algorithm should have a utc datetime"
        if tz is not None:
            dt = dt.astimezone(tz)
        return dt

    @api_method
    def set_slippage(self, us_equities=None, us_futures=None):
        """
        Set the slippage models for the simulation.

        Parameters
        ----------
        us_equities : EquitySlippageModel
            The slippage model to use for trading US equities.
        us_futures : FutureSlippageModel
            The slippage model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.slippage.SlippageModel`
        """
        if self.initialized:
            raise SetSlippagePostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type='equities',
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.slippage_models[Equity] = us_equities

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type='futures',
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.slippage_models[Future] = us_futures

    @api_method
    def set_commission(self, us_equities=None, us_futures=None):
        """Sets the commission models for the simulation.

        Parameters
        ----------
        us_equities : EquityCommissionModel
            The commission model to use for trading US equities.
        us_futures : FutureCommissionModel
            The commission model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.commission.PerShare`
        :class:`zipline.finance.commission.PerTrade`
        :class:`zipline.finance.commission.PerDollar`
        """
        if self.initialized:
            raise SetCommissionPostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type='equities',
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.commission_models[Equity] = us_equities

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type='futures',
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.commission_models[Future] = us_futures

    @api_method
    def set_cancel_policy(self, cancel_policy):
        """Sets the order cancellation policy for the simulation.

        Parameters
        ----------
        cancel_policy : CancelPolicy
            The cancellation policy to use.

        See Also
        --------
        :class:`zipline.api.EODCancel`
        :class:`zipline.api.NeverCancel`
        """
        if not isinstance(cancel_policy, CancelPolicy):
            raise UnsupportedCancelPolicy()

        if self.initialized:
            raise SetCancelPolicyPostInit()

        self.blotter.cancel_policy = cancel_policy

    @api_method
    def set_symbol_lookup_date(self, dt):
        """Set the date for which symbols will be resolved to their assets
        (symbols may map to different firms or underlying assets at
        different times)

        Parameters
        ----------
        dt : datetime
            The new symbol lookup date.
        """
        try:
            self._symbol_lookup_date = pd.Timestamp(dt, tz='UTC')
        except ValueError:
            raise UnsupportedDatetimeFormat(input=dt,
                                            method='set_symbol_lookup_date')

    # Remain backwards compatibility
    @property
    def data_frequency(self):
        return self.sim_params.data_frequency

    @data_frequency.setter
    def data_frequency(self, value):
        assert value in ('daily', 'minute')
        self.sim_params.data_frequency = value

    @api_method
    @require_initialized(HistoryInInitialize())
    def history(self, bar_count, frequency, field, ffill=True):
        """DEPRECATED: use ``data.history`` instead.
        """
        warnings.warn(
            "The `history` method is deprecated.  Use `data.history` instead.",
            category=ZiplineDeprecationWarning,
            stacklevel=4
        )

        return self.get_history_window(
            bar_count,
            frequency,
            self._calculate_universe(),
            field,
            ffill
        )

    def get_history_window(self, bar_count, frequency, assets, field, ffill):
        if not self._in_before_trading_start:
            return self.data_portal.get_history_window(
                assets,
                self.datetime,
                bar_count,
                frequency,
                field,
                self.data_frequency,
                ffill,
            )
        else:
            # If we are in before_trading_start, we need to get the window
            # as of the previous market minute
            adjusted_dt = \
                self.trading_calendar.previous_minute(
                    self.datetime
                )

            window = self.data_portal.get_history_window(
                assets,
                adjusted_dt,
                bar_count,
                frequency,
                field,
                self.data_frequency,
                ffill,
            )

            # Get the adjustments between the last market minute and the
            # current before_trading_start dt and apply to the window
            adjs = self.data_portal.get_adjustments(
                assets,
                field,
                adjusted_dt,
                self.datetime
            )
            window = window * adjs

            return window

    ####################
    # Account Controls #
    ####################

    def register_account_control(self, control):
        """
        Register a new AccountControl to be checked on each bar.
        """
        if self.initialized:
            raise RegisterAccountControlPostInit()
        self.account_controls.append(control)

    def validate_account_controls(self):
        for control in self.account_controls:
            control.validate(self.portfolio,
                             self.account,
                             self.get_datetime(),
                             self.trading_client.current_data)

    @api_method
    def set_max_leverage(self, max_leverage):
        """Set a limit on the maximum leverage of the algorithm.

        Parameters
        ----------
        max_leverage : float
            The maximum leverage for the algorithm. If not provided there will
            be no maximum.
        """
        control = MaxLeverage(max_leverage)
        self.register_account_control(control)

    @api_method
    def set_min_leverage(self, min_leverage, grace_period):
        """Set a limit on the minimum leverage of the algorithm.

        Parameters
        ----------
        min_leverage : float
            The minimum leverage for the algorithm.
        grace_period : pd.Timedelta
            The offset from the start date used to enforce a minimum leverage.
        """
        deadline = self.sim_params.start_session + grace_period
        control = MinLeverage(min_leverage, deadline)
        self.register_account_control(control)

    ####################
    # Trading Controls #
    ####################

    def register_trading_control(self, control):
        """
        Register a new TradingControl to be checked prior to order calls.
        """
        if self.initialized:
            raise RegisterTradingControlPostInit()
        self.trading_controls.append(control)

    @api_method
    def set_max_position_size(self,
                              asset=None,
                              max_shares=None,
                              max_notional=None,
                              on_error='fail'):
        """Set a limit on the number of shares and/or dollar value held for the
        given sid. Limits are treated as absolute values and are enforced at
        the time that the algo attempts to place an order for sid. This means
        that it's possible to end up with more than the max number of shares
        due to splits/dividends, and more than the max notional due to price
        improvement.

        If an algorithm attempts to place an order that would result in
        increasing the absolute value of shares/dollar value exceeding one of
        these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares to hold for an asset.
        max_notional : float, optional
            The maximum value to hold for an asset.
        """
        control = MaxPositionSize(asset=asset,
                                  max_shares=max_shares,
                                  max_notional=max_notional,
                                  on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self,
                           asset=None,
                           max_shares=None,
                           max_notional=None,
                           on_error='fail'):
        """Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares that can be ordered at one time.
        max_notional : float, optional
            The maximum value that can be ordered at one time.
        """
        control = MaxOrderSize(asset=asset,
                               max_shares=max_shares,
                               max_notional=max_notional,
                               on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_count(self, max_count, on_error='fail'):
        """Set a limit on the number of orders that can be placed in a single
        day.

        Parameters
        ----------
        max_count : int
            The maximum number of orders that can be placed on any single day.
        """
        control = MaxOrderCount(on_error, max_count)
        self.register_trading_control(control)

    @api_method
    def set_do_not_order_list(self, restricted_list, on_error='fail'):
        """Set a restriction on which assets can be ordered.

        Parameters
        ----------
        restricted_list : container[Asset], SecurityList
            The assets that cannot be ordered.
        """
        if isinstance(restricted_list, SecurityList):
            warnings.warn(
                "`set_do_not_order_list(security_lists.leveraged_etf_list)` "
                "is deprecated. Use `set_asset_restrictions("
                "security_lists.restrict_leveraged_etfs)` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2
            )
            restrictions = SecurityListRestrictions(restricted_list)
        else:
            warnings.warn(
                "`set_do_not_order_list(container_of_assets)` is deprecated. "
                "Create a zipline.finance.asset_restrictions."
                "StaticRestrictions object with a container of assets and use "
                "`set_asset_restrictions(StaticRestrictions("
                "container_of_assets))` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2
            )
            restrictions = StaticRestrictions(restricted_list)

        self.set_asset_restrictions(restrictions, on_error)

    @api_method
    @expect_types(
        restrictions=Restrictions,
        on_error=str,
    )
    def set_asset_restrictions(self, restrictions, on_error='fail'):
        """Set a restriction on which assets can be ordered.

        Parameters
        ----------
        restricted_list : Restrictions
            An object providing information about restricted assets.

        See Also
        --------
        zipline.finance.asset_restrictions.Restrictions
        """
        control = RestrictedListOrder(on_error, restrictions)
        self.register_trading_control(control)
        self.restrictions |= restrictions

    @api_method
    def set_long_only(self, on_error='fail'):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        self.register_trading_control(LongOnly(on_error))