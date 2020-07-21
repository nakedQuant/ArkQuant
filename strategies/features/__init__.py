# -*- coding :utf-8 -*-

from abc import ABC ,abstractmethod
from functools import reduce ,wraps
import pandas as pd,numpy as np

from gateWay import Event,GateReq,Quandle

def remove_na(f):
    @wraps(f)
    def wrapper(*args):
        result = f(*args)
        if isinstance(result,(pd.DataFrame,pd.Series)):
            result.dropna(inplace = True)
        return result
    return wrapper

class BaseFeature(ABC):
    """
        base feature common api
        window = 1 means return itself
    """
    _pairwise = False
    _windowed = True

    @classmethod
    def _validate_fields(cls,feed,fields):
        if isinstance(feed, dict) and set(fields) in set(list(feed.keys())):
            raise ValueError('dict feed keys must equal as fields')
        elif isinstance(feed,pd.DataFrame) and set(fields) in set(list(feed.columns)):
            raise  ValueError('dataframe columns must be same with fields')

    @classmethod
    @abstractmethod
    def calc_feature(cls,feed,window = None):
        pass


class VMA(BaseFeature):
    '''
        HF=（开盘价+收盘价+最高价+最低价）/4
        VMA指标比一般平均线的敏感度更高
    '''
    _n_fields = ['open','high','low','close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        vma = (raw['open'] + raw['high'] + raw['low'] + raw['close'])/4
        vma_windowed = vma.rolling(window = window).mean()
        return vma_windowed


class TR(BaseFeature):
    '''
        ATR又称Average true range平均真实波动范围，简称ATR指标，是由J.Welles Wilder发明的，ATR指标主要是用来衡量市场波动的强烈度，即为了显示市场变化率
        计算方法：
        1.TR =∣最高价 - 最低价∣，∣最高价 - 昨收∣，∣昨收 - 最低价∣中的最大值(np.abs(preclose.shift(1))
        2.真实波幅（ATR）= MA(TR, N)（TR的N日简单移动平均）
        3.常用参数N设置为14日或者21日
        4、atr min或者atr max真实价格变动high - low, low - prelow
    '''
    _n_fields = ['high','low','close']
    _windowed = False

    @classmethod
    @remove_na
    def calc_feature(cls,feed):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        df = pd.DataFrame(index = raw.index)
        df.loc[:,'h-l'] = abs(raw['high'] - raw['low'])
        df.loc[:,'h-c'] = abs(raw['high'] - raw['close'])
        df.loc[:,'c-l'] = abs(raw['close'] - raw['low'])
        tr = df.max(axis = 1)
        return tr


class Atr(BaseFeature):

    _n_fields = ['high','low','close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        tr = TR.calc_feature(feed)
        atr = tr.rolling(window = window).mean()
        return atr


class MA(BaseFeature):

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        ma_windowed = raw.rolling(window = window).mean()
        return ma_windowed


class EMA(BaseFeature):
    '''
        EMA,EXPMA 1 / a = 1 + (1 - a) + (1 - a) ** 2 + (1 - a) ** n ，基于等比数列，当N趋向于无穷大的 ，一般a = 2 / (n + 1)
        比如EMA10 ， 初始为10个交易日之前的数据，考虑范围放大为20，这样计算出来的指标精确度会有所提高，更加平滑
        _pairwise is True 与 _weight不能共存
    '''
    _weight = None

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window = None):
        if window is None and cls._weight is None:
            raise ValueError('when window is None ,weight must not be None')
        raw = feed.copy()
        raw.dropna(inplace= True)
        weight = 2/(window - 1) if cls._weight is None else cls._weight
        recursion = lambda x,y : x * (1- weight ) + y * weight
        reduction = [reduce(recursion,np.array(raw[:idx])) for idx in range(window,len(raw)+1)]
        res = pd.Series(reduction,index = raw.index[-len(reduction):])
        return res

class SMA(BaseFeature):
    '''
        SMA区别 wgt = (window/2 +1)/window
    '''
    _weight = None

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        weight = (window/2 + 1) / window if cls._weight is None else cls._weight
        recursion = lambda x,y : x * (1- weight ) + y * weight
        res = [reduce(recursion , np.array(raw[:idx])) for idx in range(window,len(raw)+1)]
        sma = pd.Series(res,index = raw.index[-len(res):])
        return sma

class XEMA(BaseFeature):
    """
        XEMA --- multi（X dimension) ema ,when dimension == 1  xema is EMA
        event : asset,trading_dt,field
    """
    _weight = None
    _dimension = 2

    def _recursion(self, raw,window,wgt,count = 0):
        reduction = lambda x,y : x * (1- wgt ) + y * wgt
        if count == self._dimension:
            return raw
        raw = [reduce(reduction, np.array(raw[:idx])) for idx in range(window,len(raw)+1)]
        count = count +1
        return self._recursion(raw,window,wgt,count)

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        weight = 2 / (window - 1) if cls._weight is None else cls._weight
        instance = cls()
        data = instance._recursion(raw,window,weight)
        xema = pd.Series(data,index = raw.index[-len(data):])
        return xema


class AEMA(BaseFeature):
    """
        AEMA is distinct from ema where weight is not the same and changes according to the time
        wgt : List e.g. 1- wgt , wgt ,but wgt is changeable
        raw : list or array
        return aema value

    """
    _weight = None
    _windowed = False

    @classmethod
    @remove_na
    def calc_feature(cls,feed):
        raw = feed.copy()
        align_wgt = list(cls._weight[:len(raw) - 1])
        align_copy = align_wgt.copy()
        align_copy.append(0)
        align_copy.reverse()
        align_wgt.reverse()
        align_wgt.append(1)
        ratio = np.cumprod(1 - np.array(align_copy)) * np.array(align_wgt)
        aema = (np.array(raw) * ratio).sum()
        return aema


class EWMA(BaseFeature):
    """
        区别与EMA ，权重为（1-a) ** i 按照时间的推移权重指数变化
    """
    _weight = None

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        if not cls._weight:
            wgt  = [(1 - 2/(window + 1)) ** n for n in range(len(raw))]
        else:
            wgt  = [(1 - cls._weight) ** n for n in range(len(raw))]
        AEMA._weight = wgt
        data = [AEMA.calc_feature(raw[:idx]) for idx in range(window,len(raw)+1)]
        ewma_windowed = pd.Series(data,index = raw.index[-len(data):])
        return ewma_windowed


if __name__ == '__main__':

    date = '2019-06-01'
    asset = '000001'
    window = 60
    fields = ['close']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)
    # # tr=TR.calc_feature(feed)
    # # print('tr',tr)
    # # atr = Atr.calc_feature(feed,5)
    # # print('atr',atr)
    # # ma = MA.calc_feature(feed,5)
    # # print('ma',ma)
    # print('feed',feed)
    ema = EMA.calc_feature(feed,5)
    print('ema',ema)
    # sma = SMA.calc_feature(feed,5)
    # print('sma',sma)
    xema = XEMA.calc_feature(feed,5)
    print('xema',xema)
    # # ewma = EWMA.calc_feature(feed,5)
    # # print('ewma',ewma)
