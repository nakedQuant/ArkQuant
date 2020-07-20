#计算系数
from functools import partial
import pandas as pd
from abc import ABC,abstractmethod


class HistoryCompatibleAdjustments(object):
    """
        calculate adjustments coef
    """
    def __init__(self,
                 _reader,
                 adjustment_reader
                 ):
        self._adjustments_reader = adjustment_reader
        self._reader = _reader

    def _init_raw_array(self,assets,edate,window):
        if self._reader.data_frequency == 'daily':
            bars = self._reader.load_raw_arrays(edate, window, ['close'], assets)
        elif self._reader.data_frequency == 'minute':
            bars = self._reader.load_ticker_arrays(edate,window,assets,['close'],'15:00')
        return bars

    @staticmethod
    def _calculate_divdends_for_sid(adjustment,close,sid):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        bar = close[sid]
        try:
            divdend_fq = adjustment['divdends'][sid]
            exdate_close = bar['close'][bar['trade_dt'] == divdend_fq['ex_date']]
            qfq = (1- divdend_fq['bonus'] /
                          (10 * exdate_close)) / \
                         (1 + (divdend_fq['sid_bonus'] +
                               divdend_fq['sid_transfer']) / 10)
        except KeyError:
            qfq = pd.Series(1.0,index = bar['trade_dt'])
        return qfq

    @staticmethod
    def _calculate_rights_for_sid(adjustment,close,sid):
        """
           配股除权价=（除权登记日收盘价+配股价*每股配股比例）/（1+每股配股比例）
        """
        bar = close[sid]
        try:
            rights_fq = adjustment['rights'][sid]
            exdate_close = bar['close'][bar['trade_dt'] == rights_fq['ex_date']]
            qfq = (exdate_close + (rights_fq['rights_price'] *
                       rights_fq['rights_bonus']) / 10) \
                 / (1 + rights_fq['rights_bonus']/10)
        except KeyError:
            qfq = pd.Series(1.0,index = bar['trade_dt'])
        return qfq

    def _calculate_adjustments_for_sid(self,adjustment,close,sid):
        fq = self._calculate_divdends_for_sid(adjustment,close,sid)
        fq_rights = self._calculate_rights_for_sid(adjustment,close,sid)
        fq.append(fq_rights)
        fq.sort_index(ascending= False,inplace = True)
        qfq = 1 / fq.cumprod()
        return qfq

    def calculate_adjustments_in_sessions(self,edate,window,assets):
        """
        Returns
        -------
        adjustments : list[dict[int -> Adjustment]]
            A list, where each element corresponds to the `columns`, of
            mappings from index to adjustment objects to apply at that index.
        """
        adjs = {}
        #获取全部的分红除权配股数据
        adjustments = self._adjustments_reader.load_pricing_adjustments(edate,window)
        #获取对应的收盘价数据
        close_bars = self._init_raw_array(edate,window,assets)
        #计算前复权系数
        _calculate = partial(self._calculate_adjustments_for_sid,adjustments = adjustments,close = close_bars)
        for asset in assets:
            adjs[asset] = _calculate(sid = asset.sid)
        return adjs


class SlidingWindow(ABC):

    @property
    def frequency(self):
        raise ValueError()

    @abstractmethod
    def _array(self):
        raise NotImplementedError()

    @abstractmethod
    def _window_arrays(self):
        raise NotImplementedError()


#sliding window
class AdjustedDailyWindow(SlidingWindow):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """
    FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])

    def __init__(self,
                trading_calendar,
                bar_reader,
                equity_adjustment_reader):
        self._trading_calendar = trading_calendar
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    bar_reader)

    @property
    def frequency(self):
        return 'daily'

    def _array(self, dts, assets, field):
        _reader =  self._adjustment._reader
        raw = _reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )
        return raw

    def _window_arrays(self,edate,window,assets,field):
        """基于固定的fields才需要adjust"""
        #获取时间区间
        session = self._trading_calendar.sessions_in_range(edate,window)
        # 获取原始数据
        raw_arrays = self._array(session,assets, field)
        #需要调整的
        adjusted_fields = set(field) & self.FIELDS
        if adjusted_fields:
            #调整系数
            adjustments = self._adjustment.calculate_adjustments_in_sessions(edate,window,assets)
            #计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                qfq = adjustments[sid]
                raw = raw_arrays[sid]
                try:
                    qfq = qfq.reindex(session)
                    qfq.fillna(method = 'bfill',inplace = True)
                    qfq.fillna(1.0,inplace = True)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                except Exception as e:
                    print(e,asset)
                adjust_arrays[sid] = raw
        else:
            adjust_arrays = raw_arrays
        return adjust_arrays


class AdjustedMinuteWindow(SlidingWindow):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """
    FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])

    def __init__(self,
                _minute_reader,
                equity_adjustment_reader,
                trading_calendar):
        self._trading_calendar = trading_calendar
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    _minute_reader)

    @property
    def frequency(self):
        return 'minute'

    def _array(self, dts, assets, field):
        _reader = self._adjustment._reader
        raw = _reader.load_raw_arrays(
                                    dts[0],
                                    dts[-1],
                                    assets,
                                    field
        )
        return raw

    def _window_arrays(self,edate,window,assets,field):
        """基于固定的fields才需要adjust"""
        #获取时间区间
        session = self._trading_calendar.sessions_in_range(edate,window)
        # 获取原始数据
        raw_arrays = self._array(session,assets,field)
        #需要调整的
        adjusted_fields = set(field) & self.FIELDS
        if adjusted_fields:
            #调整系数
            adjustments = self._adjustment.calculate_adjustments_in_sessions(edate,window,assets)
            #计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                #调整index
                qfq = adjustments[sid]
                qfq.index = [ pd.Timestamp(inx).timestamp() + 15 * 60 * 60  for inx in qfq.index]
                raw = raw_arrays[sid]
                try:
                    qfq = qfq.reindex(session)
                    qfq.fillna(method = 'bfill',inplace = True)
                    qfq.fillna(1.0,inplace = True)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                except Exception as e:
                    print(e,asset)
                adjust_arrays[sid] = raw
        else:
            adjust_arrays = raw_arrays
        return adjust_arrays

__all__ = [AdjustedMinuteWindow,AdjustedDailyWindow,HistoryCompatibleAdjustments]