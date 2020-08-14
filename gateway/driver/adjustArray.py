# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from functools import partial, lru_cache
import pandas as pd
from toolz import valmap

__all__ = [
    'AdjustedMinuteWindow',
    'AdjustedDailyWindow',
    'HistoryCompatibleAdjustments'
]


class HistoryCompatibleAdjustments(object):
    """
        calculate adjustments coef
    """
    def __init__(self,
                 adjustment_reader,
                 reader,
                 ):
        self._adjustments_reader = adjustment_reader
        self._reader = reader

    @property
    def reader(self):
        return self._reader

    @property
    def data_frequency(self):
        return self._reader.data_frequency

    # @lru_cache(maxsize=8)
    # def _load_raw_array(self, asset, date, window):
    #     sessions = _calendar.session_in_window(date, window, include=True)
    #     close = self.reader.load_raw_arrays(sessions, asset, ['open', 'high', 'low', 'close', 'volume', 'amount'])
    #     return close, sessions

    @lru_cache(maxsize=8)
    def _load_raw_array(self, assets, sessions):
        # sessions = _calendar.session_in_window(date, window, include=True)
        close = self.reader.load_raw_arrays(sessions, assets, ['open', 'high', 'low', 'close', 'volume', 'amount'])
        return close, sessions

    @staticmethod
    def _calculate_divdends_for_sid(adjustment, close, sid):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        bundles = close[sid]
        try:
            divdends = adjustment['divdends'][sid]
            ex_close = bundles['close'][bundles['trade_dt'] == divdends['ex_date']]
            qfq = (1 - divdends['bonus']/(10 * ex_close)) / \
                  (1 + (divdends['sid_bonus'] + divdends['sid_transfer']) / 10)
        except KeyError:
            qfq = pd.Series(1.0, index=bundles['trade_dt'])
        return qfq

    @staticmethod
    def _calculate_rights_for_sid(adjustment, close, sid):
        """
           配股除权价=（除权登记日收盘价+配股价*每股配股比例）/（1+每股配股比例）
        """
        bundles = close[sid]
        try:
            rights = adjustment['rights'][sid]
            ex_close = bundles['close'][bundles['trade_dt'] == rights['ex_date']]
            qfq = (ex_close + (rights['rights_price'] * rights['rights_bonus']) / 10) / \
                  (1 + rights['rights_bonus']/10)
        except KeyError:
            qfq = pd.Series(1.0, index=bundles['trade_dt'])
        return qfq

    def _calculate_adjustments_for_sid(self, adjustment, close, sid):
        fq_divdends = self._calculate_divdends_for_sid(adjustment, close, sid)
        fq_rights = self._calculate_rights_for_sid(adjustment, close, sid)
        fq = fq_divdends.append(fq_rights)
        fq.sort_index(ascending=False, inplace=True)
        qfq = 1 / fq.cumprod()
        return qfq

    def adapt_to_frequency(self, adjustments):
        if self.data_frequency == 'minute':
            adaption = adjustments.rename(index=lambda x: pd.Timestamp(x).timestamp() + 15 * 60 * 60, inplace=True)
        else:
            adaption = adjustments
        return adaption

    def calculate_adjustments_in_sessions(self, sessions, assets):
        """
        Returns
        -------
        adjustments : list[dict[int -> Adjustment]]
            A list, where each element corresponds to the `columns`, of
            mappings from index to adjustment objects to apply at that index.
        """
        adjs = {}
        # 获取全部的分红除权配股数据
        adjustments = self._adjustments_reader.load_pricing_adjustments(sessions)
        # 基于data_frequency --- 调整adjustments
        adapted_adjustments = self.adapt_to_frequency(adjustments)
        # 获取对应的收盘价数据
        history, sessions = self._load_raw_array(sessions, assets)
        close = valmap(lambda x: x['close'], history)
        # 计算前复权系数
        _calculate = partial(self._calculate_adjustments_for_sid, adjustments=adapted_adjustments, close=close)
        for asset in assets:
            adjs[asset] = _calculate(sid=asset.sid)
        return adjs, history, sessions


class SlidingWindow(object):

    FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])

    @property
    def frequency(self):
        return None

    @property
    def reader(self):
        return self._adjustment.reader

    def array(self, dts, assets, fields):
        """
        :param dts:  list (length 2)
        :param assets: list
        :param fields: list
        :return: unadjusted data
        """
        original = self.reader.load_raw_arrays(
            dts,
            assets,
            fields
        )
        return original

    def window_arrays(self, sessions, assets, field):
        """
        :param sessions: [a,b]
        :param assets: Assets list
        :param field: str or list
        :return: arrays which is adjusted by divdends and rights
        """
        adjustments, raw_arrays, sessions = self._adjustment.calculate_adjustments_in_sessions(sessions, assets)
        adjusted_fields = set(field) & self.FIELDS
        if adjusted_fields:
            #计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                qfq = adjustments[sid]
                raw = raw_arrays[sid]
                try:
                    qfq = qfq.reindex(sessions)
                    qfq.fillna(method='bfill', inplace=True)
                    qfq.fillna(1.0, inplace=True)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                except Exception as e:
                    print(e, asset)
                adjust_arrays[sid] = raw
        else:
            adjust_arrays = raw_arrays
        return adjust_arrays


class AdjustedDailyWindow(SlidingWindow):
    """
        Wrapper around an AdjustedArrayWindow which supports monotonically
        increasing (by datetime) requests for a sized window of data.
    """
    def __init__(self,
                 bar_reader,
                 equity_adjustment_reader):
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    bar_reader)

    @property
    def frequency(self):
        return 'daily'


class AdjustedMinuteWindow(SlidingWindow):
    """
        Wrapper around an AdjustedArrayWindow which supports monotonically
        increasing (by datetime) requests for a sized window of data.
    """
    def __init__(self,
                 minute_reader,
                 equity_adjustment_reader):
        self._adjustment = HistoryCompatibleAdjustments(
                                    equity_adjustment_reader,
                                    minute_reader)

    @property
    def frequency(self):
        return 'minute'
