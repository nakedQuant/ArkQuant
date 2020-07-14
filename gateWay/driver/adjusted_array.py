# -*- coding :ur¥tf-8 -*-

import pandas as pd,datetime as dt
from .adjustments import  SQLiteAdjustmentReader
from .bar import BarReader
from .trading_calendar import TradingCalendar


class AdjustedArray(object):
    """
        1. adjustments raw_arrays
        2. update --- adjustment
    """
    ADJUST_FIELDS = frozenset([
        "open",
        "high",
        "low",
        "close",
        "volume",
    ])

    def __init__(self):
        self._reader = {
                    'bar': BarReader(),
                    'adjustment': SQLiteAdjustmentReader(),
                    }
        self.trading_calendar = TradingCalendar()
        self._cache_adjustments()

    def _cache_adjustments(self):
        self.adjustment_cache = self.reader['adjustment'].load_pricing_adjustments()

    @staticmethod
    def _calculate_divdends_for_sid(adjust,close,date):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        divdend = adjust['divdends']
        divdend_fq = divdend[divdend['record_date'] <= date]
        record_close = close['close'][close['trade_dt'] == divdend_fq['record_date']]
        qfq = (1- divdend['bonus'] /
                      (10 * record_close)) / \
                     (1 + (divdend_fq['sid_bonus'] +
                           divdend_fq['sid_transfer']) / 10)
        return qfq

    @staticmethod
    def _calculate_rights_for_sid(adjust,bar,date):
        """
           配股除权价=（除权登记日收盘价+配股价*每股配股比例）/（1+每股配股比例）
        """
        rights = adjust['rights']
        rights_fq = rights[rights['record_date'] <= date]
        record_close = bar['close'][bar['trade_dt'] == rights_fq['record_date']]
        fq = (record_close + (rights_fq['rights_price'] *
                   rights_fq['rights_bonus']) / 10) \
             / (1 + rights_fq['rights_bonus']/10)
        return fq

    def _calculate_adjust_for_sid(self,adjustment,kline,date):
        fq = self._calculate_divdends_for_sid(adjustment,kline,date)
        fq_rights = self._calculate_rights_for_sid(adjustment,kline,date)
        fq.append(fq_rights)
        fq.sort_index(ascending= False,inplace = True)
        qfq = 1 / fq.cumprod()
        return qfq

    def _calculate_impl(self,edate,window,fields,assets):
        trading_calendar = self.trading_calendar.session_in_range_window(edate,window)
        adjustments = self.adjustment_cache
        adjusted_sid_fq = dict()

        expected_fields = set(fields) & self.ADJUST_FIELDS
        assert expected_fields,ValueError('expected fields are invalid')
        if assets:
            unpack_kline = self._reader['bar'].load_asset_kline(edate,
                                                        window,
                                                        fields,
                                                        assets = assets)
        else:
            #只有股票数据需要基于分红除权进行更新,symbol
            unpack_kline = self._reader['bar'].load_asset_kline(edate,
                                                        window,
                                                        fields,
                                                        category = 'symbol')
        for sid,sid_kline in unpack_kline.items():
            try:
                sid_adjustment = adjustments[sid]
                fq = self._calculate_adjust_for_sid(
                    sid_adjustment,
                    sid_kline,
                    edate
                )
                fq = fq.reindex(trading_calendar)
                fq.fillna(method = 'bfill',inplace = True)
                fq.fillna(1.0,inplace = True)
            except KeyError:
                fq = pd.Series(1.0,index = trading_calendar )
        adjusted_sid_fq[sid] = fq
        return adjusted_sid_fq, unpack_kline, expected_fields

    def load_adjusted_array(self,edate,window,fields,assets = None):
        """
            return adjusted_arrays
        """
        adjust_qfq,unpack_kline,adjusted_fields = \
            self._calculate_impl(
            edate,
            window,
            fields,
            assets)

        adjust_array = dict()
        for sid , sid_kline in unpack_kline.items():
            sid_fq = adjust_qfq[sid]
            sid_kline[adjusted_fields] = sid_kline[adjusted_fields].multiply(
                sid_fq,axis = 0)
            adjust_array[sid] = sid_kline
        return adjust_array

    def load_raw_array(self,edate,window,fields,_types):
        """获取大类标的数据"""
        adjust_array = dict()
        for _typ in _types:
            raw = self._reader['bar'].load_asset_kline(edate,
                                                         window,
                                                         fields,
                                                         category = _typ)
            adjust_array.update(raw)
        return adjust_array


    def load_array_for_sids(self,edate,window,fields,assets):
        """获取标的的原始数据"""
        raw_array = self._reader['bar'].load_asset_kline(
                                                    edate,
                                                    window,
                                                    fields,
                                                    assets = assets
                                                    )
        return raw_array

    def load_minutes_for_sids(self,sids,ticker = None):
        minutes = {}
        struct_time = dt.datetime.now()
        if struct_time.hour >= 9 and struct_time.minute >= 25:
            for sid in sids:
                raw = self._reader['bar'].load_minutes_kline(sid,0)
                minutes[sid] = raw.loc[ticker] if ticker else raw
        return minutes
