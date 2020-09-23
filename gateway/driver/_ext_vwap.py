#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019
@author: python
"""
import datetime, pandas as pd
from toolz import valmap
from gateway.driver.bcolz_reader import BcolzMinuteReader
from _calendar.trading_calendar import calendar


class VWAP(object):

    def __init__(self):
        self._reader = BcolzMinuteReader()

    @staticmethod
    def _calculate_vwap(frame):
        frame.loc[:, 'trade_dt'] = pd.DatetimeIndex([datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%d')
                                        for i in frame.index])
        # vwap = frame.resample('D').apply(lambda x: x['close'] * x['volume'] / x['volume'].sum())
        vwap = frame.groupby(by='trade_dt').apply(lambda x: (x['close'] * x['volume'] / x['volume'].sum()).sum())
        return vwap

    def _retrieve_minutes(self, session, asset):
        dct = self._reader.load_raw_arrays([min(session), max(session)], asset,
                                           ['open', 'high', 'low', 'close', 'volume'])
        vwap = valmap(self._calculate_vwap, dct)
        return vwap

    def calculate(self, date, window, assets):
        assets = tuple(assets)
        session = calendar.session_in_window(date, window)
        # print('session', session)
        vwap = self._retrieve_minutes(session, assets)
        return vwap


# if __name__ == '__main__':
#
#     equity = Equity('600000')
#     minute_vwap = VWAP()
#     vwap = minute_vwap.calculate('2005-09-08', 50, [equity])
#     print('vwap', vwap)
