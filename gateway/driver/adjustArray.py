# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, time
from toolz import valmap
from functools import partial
from gateway.driver.bar_reader import AssetSessionReader
from gateway.driver.bcolz_reader import BcolzMinuteReader
from gateway.driver.adjustment_reader import SQLiteAdjustmentReader
from gateway.asset.assets import Equity, Convertible, Fund


__all__ = [
    'AdjustedMinuteWindow',
    'AdjustedDailyWindow',
    'HistoryCompatibleAdjustments'
]

AdjustFields = frozenset(['open', 'high', 'low', 'close', 'volume'])


class HistoryCompatibleAdjustments(object):
    """
        calculate adjustments coef
    """
    def __init__(self,
                 reader,
                 adjustment_reader
                 ):
        self._reader = reader
        self._adjustments_reader = adjustment_reader

    @property
    def reader(self):
        return self._reader

    @property
    def data_frequency(self):
        return self._reader.data_frequency

    @staticmethod
    def _calculate_divdends_for_sid(adjustment, data, sid):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        kline = data[sid]
        kline.index = [time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(i)) for i in data['600000'].index]
        print('kline', kline)
        try:
            divdends = adjustment['divdends'][sid]
            print('union', set(divdends.index) & set(kline.index))
            ex_close = kline['close'].reindex(index=divdends.index)
            qfq = (1 - divdends['bonus']/(10 * ex_close)) / \
                  (1 + (divdends['sid_bonus'] + divdends['sid_transfer']) / 10)
        except KeyError:
            qfq = pd.Series(dtype=float)
        return qfq

    @staticmethod
    def _calculate_rights_for_sid(adjustment, data, sid):
        """
           配股除权价=（除权登记日收盘价+配股价*每股配股比例）/（1+每股配股比例）
        """
        kline = data[sid]
        try:
            rights = adjustment['rights'][sid]
            ex_close = kline['close'].reindex(index=rights.index)
            qfq = (ex_close + (rights['rights_price'] * rights['rights_bonus']) / 10) / \
                  (1 + rights['rights_bonus']/10)
        except KeyError:
            qfq = pd.Series(dtype=float)
        return qfq

    def calculate_adjustments_for_sid(self, adjustment, data, sid):
        fq_divdends = self._calculate_divdends_for_sid(adjustment, data, sid)
        # print('divdends', fq_divdends)
        fq_rights = self._calculate_rights_for_sid(adjustment, data, sid)
        # print('rights', fq_rights)
        fq = fq_divdends.append(fq_rights)
        fq.sort_index(ascending=False, inplace=True)
        qfq = 1 / fq.cumprod()
        print('qfq', qfq)
        return qfq

    def _adjust_by_frequency(self, adjustments):
        # print('adjustments', adjustments)
        if self.data_frequency == 'minute':
            def reformat(frame):
                frame.index = [int(pd.Timestamp(i).timestamp() + 15 * 60 * 60) for i in frame.index]
                return frame
            adjustments['divdends'] = valmap(lambda x: reformat(x), adjustments['divdends'])
            adjustments['rights'] = valmap(lambda x: reformat(x), adjustments['rights'])
        return adjustments

    def calculate_adjustments_in_sessions(self, sessions, assets):
        """
        Returns
        -------
        adjustments : list[dict[int -> Adjustment]]
            A list, where each element corresponds to the `columns`, of
            mappings from index to adjustment objects to apply at that index.
        sessions : list , eg['2020-01-30', '2020-08-30']

        assets:
        """
        adjs = {}
        # 获取全部的分红除权配股数据
        adjustments = self._adjustments_reader.load_pricing_adjustments(sessions)
        # 基于data_frequency --- 调整adjustments
        adapted_adjustments = self._adjust_by_frequency(adjustments)
        # 获取对应的收盘价数据
        data = self.reader.load_raw_arrays(sessions, assets, ['open', 'high', 'low', 'close', 'volume', 'amount'])
        # 计算前复权系数
        _calculate = partial(self.calculate_adjustments_for_sid, adjustment=adapted_adjustments, data=data)
        for asset_obj in assets:
            sid = asset_obj.sid
            try:
                adjs[sid] = _calculate(sid=sid)
            except KeyError:
                print('code: %s has not kline between session' % sid)
        return adjs, data

    def array(self, dts, assets, fields):
        """
        :param dts:  list (length 2)
        :param assets: list
        :param fields: list
        :return: unadjusted data
        """
        original = self._reader.load_raw_arrays(
            dts,
            assets,
            fields
        )
        return original


class SlidingWindow(object):

    @property
    def frequency(self):
        return None

    @property
    def reader(self):
        return self._compatible_adjustment.reader

    def get_spot_value(self, dt, asset, fields):
        spot_value = self.reader.get_spot_value(dt, asset, fields)
        return spot_value

    def get_stack_value(self, tbl, session):
        stack = self.reader.get_stack_value(tbl, session)
        return stack

    def array(self, dts, assets, fields):
        """
        :param dts:  list (length 2)
        :param assets: list
        :param fields: list
        :return: unadjusted data
        """
        _array = self.reader.load_raw_arrays(
            dts,
            assets,
            fields
        )
        return _array

    def window_arrays(self, sessions, assets, field):
        """
        :param sessions: [a,b]
        :param assets: Assets list
        :param field: str or list
        :return: arrays which is adjusted by divdends and rights
        """
        adjustments, raw_arrays = self._compatible_adjustment.calculate_adjustments_in_sessions(sessions, assets)
        adjusted_fields = list(set(field) & AdjustFields)
        if adjusted_fields:
            # 计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                try:
                    raw = raw_arrays[sid]
                    qfq = adjustments[sid]
                    qfq = qfq.reindex(index=set(raw.index))
                    qfq.sort_index(inplace=True)
                    qfq.fillna(method='bfill', inplace=True)
                    qfq.fillna(1.0, inplace=True)
                    # print('full qfq', qfq)
                    raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
                    adjust_arrays[sid] = raw[adjusted_fields]
                    # print('adjust_raw', raw[adjusted_fields])
                except KeyError:
                    adjust_arrays[sid] = pd.DataFrame()
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
        self._compatible_adjustment = HistoryCompatibleAdjustments(
                                    bar_reader,
                                    equity_adjustment_reader,
                              )

    @property
    def frequency(self):
        return 'daily'

    def get_equity_pctchange(self, dts):
        frame_pct = self.reader.get_equity_pctchange(dts)
        return frame_pct


class AdjustedMinuteWindow(SlidingWindow):
    """
        Wrapper around an AdjustedArrayWindow which supports monotonically
        increasing (by datetime) requests for a sized window of data.
    """
    def __init__(self,
                 minute_reader,
                 equity_adjustment_reader):
        self._compatible_adjustment = HistoryCompatibleAdjustments(
                                    minute_reader,
                                    equity_adjustment_reader
                            )

    @property
    def frequency(self):
        return 'minute'


if __name__ == '__main__':

    minute_reader = BcolzMinuteReader()
    session_reader = AssetSessionReader()
    adjust_reader = SQLiteAdjustmentReader()

    asset = Equity('600000')
    sessions = ['2005-01-10', '2005-01-11']
    fields = ['open', 'close']
    # his = HistoryCompatibleAdjustments(session_reader, adjust_reader)
    # his.calculate_adjustments_in_sessions(['2017-08-10', '2020-10-30'], [asset])
    # window_arrays = his.window_arrays(['2010-08-10', '2015-10-30'], [asset], ['open', 'close'])
    # print('window_arrays', window_arrays)
    # original = his.array(sessions, [asset], ['open', 'close'])
    # print('original array', original)

    # daily_adjust = AdjustedDailyWindow(session_reader, adjust_reader)
    # close = daily_adjust.window_arrays(sessions, [asset], ['close'])
    # print('daily adjust close', close)
    # raw_close = daily_adjust.array(sessions, [asset], ['close'])
    # print('raw_close', raw_close)

    minute_adjust = AdjustedMinuteWindow(minute_reader, adjust_reader)
    # minute_spot = minute_adjust.get_spot_value('2020-09-03', asset, fields)
    # print('minute spot value', minute_spot)
    # minute_array = minute_adjust.array(sessions, [asset], fields)
    # print('minute_array', minute_array)
    minute_window_array = minute_adjust.window_arrays(sessions, [asset], fields)
    print('minute_window_array', minute_window_array)
