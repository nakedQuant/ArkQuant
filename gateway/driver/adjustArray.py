# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from toolz import valmap
from functools import partial

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

    def _adapt_by_frequency(self, adjustments):
        # print('adjustments', adjustments)
        if self.data_frequency == 'minute':
            def reformat(frame):
                # minutes --- 14:29
                frame.index = [int(pd.Timestamp(i).timestamp() + 15 * 60 * 60 - 60) for i in frame.index]
                return frame
            adjustments['dividends'] = valmap(lambda x: reformat(x), adjustments['dividends'])
            adjustments['rights'] = valmap(lambda x: reformat(x), adjustments['rights'])
        return adjustments

    @staticmethod
    def _calculate_dividends_for_sid(adjustment, data, sid):
        """
           股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
           前复权：复权后价格=(复权前价格-现金红利)/(1+流通股份变动比例)
           后复权：复权后价格=复权前价格×(1+流通股份变动比例)+现金红利
        """
        # kline.index = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M') for i in data['600000'].index]
        kline = data[sid]
        try:
            dividends = adjustment['dividends'][sid]
            # print('dividends union', set(dividends.index) & set(kline.index))
            ex_close = kline['close'].reindex(index=dividends.index)
            qfq = (1 - dividends['bonus']/(10 * ex_close)) / \
                  (1 + (dividends['sid_bonus'] + dividends['sid_transfer']) / 10)
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
            print('rights', sid, rights)
            ex_close = kline['close'].reindex(index=rights.index)
            print('ex_close', ex_close)
            qfq = (ex_close + (rights['rights_price'] * rights['rights_bonus']) / 10) / \
                  (1 + rights['rights_bonus']/10)
        except KeyError:
            qfq = pd.Series(dtype=float)
        return qfq

    def calculate_coef_for_sid(self, adjustment, data, sid):
        fq_dividends = self._calculate_dividends_for_sid(adjustment, data, sid)
        fq_rights = self._calculate_rights_for_sid(adjustment, data, sid)
        # print('rights', fq_rights)
        fq = fq_dividends.append(fq_rights)
        fq.sort_index(ascending=False, inplace=True)
        qfq = 1 / fq.cumprod()
        # print('qfq', qfq)
        return qfq

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
        adapted_adjustments = self._adapt_by_frequency(adjustments)
        # 获取对应的收盘价数据
        data = self.reader.load_raw_arrays(sessions, assets, ['open', 'high', 'low', 'close', 'volume', 'amount'])
        # 计算前复权系数
        _calculate = partial(self.calculate_coef_for_sid, adjustment=adapted_adjustments, data=data)
        for asset_obj in assets:
            sid = asset_obj.sid
            try:
                adjs[sid] = _calculate(sid=sid)
            except KeyError:
                print('code: %s has not kline between session' % sid)
        return adjs, data


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

    def get_stack_value(self, tbl, sessions):
        stack = self.reader.get_stack_value(tbl, sessions)
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
        adjustments, frame_dcts = self._compatible_adjustment.calculate_adjustments_in_sessions(sessions, assets)
        adjusted_fields = list(set(field) & AdjustFields)
        if adjusted_fields:
            # 计算调整数据
            adjust_arrays = {}
            for asset in assets:
                sid = asset.sid
                try:
                    frame = frame_dcts[sid]
                    qfq = adjustments[sid]
                    qfq = qfq.reindex(index=set(frame.index))
                    qfq.sort_index(inplace=True)
                    qfq.fillna(method='bfill', inplace=True)
                    qfq.fillna(1.0, inplace=True)
                    # print('final qfq coef', qfq)
                    frame[adjusted_fields] = frame.loc[:, adjusted_fields].multiply(qfq, axis=0)
                    # adjust_arrays[sid] = frame[adjusted_fields]
                    adjust_arrays[sid] = frame
                except KeyError:
                    adjust_arrays[sid] = pd.DataFrame()
        else:
            adjust_arrays = frame_dcts

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

    def get_mkv_value(self, session, assets, fields):
        mkv = self.reader.get_mkv_value(session, assets, fields)
        return mkv


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


__all__ = [
    'AdjustedMinuteWindow',
    'AdjustedDailyWindow',
    'HistoryCompatibleAdjustments'
]

# if __name__ == '__main__':

#     from gateway.driver.bar_reader import AssetSessionReader
#     from gateway.driver.bcolz_reader import BcolzMinuteReader
#     from gateway.driver.adjustment_reader import SQLiteAdjustmentReader
#     from gateway.asset.assets import Equity, Convertible, Fund
#
#     minute_reader = BcolzMinuteReader()
#     session_reader = AssetSessionReader()
#     adjust_reader = SQLiteAdjustmentReader()
#
#     asset = Equity('600000')
#     sessions = ['2005-01-10', '2010-01-11']
#     fields = ['open', 'close']
#     his_daily = HistoryCompatibleAdjustments(session_reader, adjust_reader)
#     adjs, data = his_daily.calculate_adjustments_in_sessions(sessions, [asset])
#     print('adj daily coef', adjs)
#     print('daily data', data)
#
#     his_minute = HistoryCompatibleAdjustments(minute_reader, adjust_reader)
#     adjs, data = his_minute.calculate_adjustments_in_sessions(sessions, [asset])
#     print('adj minute coef', adjs)
#     print('minute data', data)
#
#     daily_sliding_window = AdjustedDailyWindow(session_reader, adjust_reader)
#     daily_spot_value = daily_sliding_window.get_spot_value('2020-09-03', asset, fields)
#     print('daily spot value', daily_spot_value)
#     daily_sliding = daily_sliding_window.window_arrays(sessions, [asset], fields)
#     print('daily_sliding ', daily_sliding)
#     daily_array = daily_sliding_window.array(sessions, [asset], fields)
#     print('daily array', daily_array)
#
#     minute_sliding_window = AdjustedMinuteWindow(minute_reader, adjust_reader)
#     minute_spot = minute_sliding_window.get_spot_value('2005-09-07', asset, fields)
#     print('minute spot value', minute_spot)
#     minute_array = minute_sliding_window.array(sessions, [asset], fields)
#     print('minute_array', minute_array)
#     minute_window_array = minute_sliding_window.window_arrays(sessions, [asset], fields)
#     print('minute_window_array', minute_window_array)
