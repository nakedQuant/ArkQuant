# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, json
from gateway.driver.tools import _parse_url
from gateway.driver.client import tsclient
from gateway.driver.resample import Freq
from gateway.driver.bar_reader import AssetSessionReader
from gateway.driver.bcolz_reader import BcolzMinuteReader
from gateway.driver.adjustment_reader import SQLiteAdjustmentReader
from gateway.driver.history import (
    HistoryDailyLoader,
    HistoryMinuteLoader
)


class DataPortal(object):
    """Interface to all of the data that a ArkQuant needs.

    This is used by the ArkQuant runner to answer questions about the data,
    like getting the prices of asset on a given day or to service history
    calls.

    Parameters
    ----------
    rule --- resample rule
    asset_finder : assets.assets.AssetFinder
        The AssetFinder instance used to resolve asset.
    """
    OHLCV_FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume', 'amount'])

    def __init__(self):
        _minute_reader = BcolzMinuteReader()
        _session_reader = AssetSessionReader()

        self._adjustment_reader = SQLiteAdjustmentReader()

        _history_daily_loader = HistoryDailyLoader(
            _session_reader,
            self._adjustment_reader,
        )
        _history_minute_loader = HistoryMinuteLoader(
            _minute_reader,
            self._adjustment_reader,

        )
        self._history_loader = {
            'daily': _history_daily_loader,
            'minute': _history_minute_loader,
        }
        self.freq_rule = Freq()
        self._extra_source = None

    @property
    def adjustment_reader(self):
        return self._adjustment_reader

    def get_dividends(self, assets, trading_day):
        """
        splits --- divdends

        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        assets: Asset list
            The asset whose stock dividends should be returned.

        trading_day: pd.DatetimeIndex
            The trading day.

        Returns
        -------
            equity divdends or cash divdends
        """
        dividends = self._adjustment_reader.retrieve_pay_date_dividends(assets, trading_day)
        return dividends

    def get_rights(self, assets, trading_day):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        assets: Asset list
            The asset whose stock dividends should be returned.

        trading_day: pd.DatetimeIndex
            The trading dt.

        Returns
        -------
            equity rights
        """
        rights = self._adjustment_reader.retrieve_ex_date_rights(assets, trading_day)
        return rights

    def get_mkv_value(self, sessions, assets, fields=None):
        mkv = self._history_loader['daily'].get_mkv_value(sessions, assets, fields)
        return mkv

    def get_spot_value(self, dts, asset, frequency, field):
        spot_value = self._history_loader[frequency].get_spot_value(dts, asset, field)
        return spot_value

    def get_stack_value(self, tbl, dt, length, frequency):
        stack = self._history_loader[frequency].get_stack_value(tbl, dt, length)
        return stack

    def get_open_pct(self, asset, dt):
        if asset.asset_type == 'equity':
            # 存在0.1%误差
            spot_value = self.get_spot_value(dt, asset, 'daily', ['open', 'high', 'low', 'close', 'pct'])
            preclose = (spot_value['high'] - spot_value['low']) * 100 / spot_value['pct']
            open_pct = spot_value['open'] / preclose - 1
        else:
            spot_value = self.get_spot_value(dt, asset, 'daily', ['open', 'close'])
            pre_value = self.get_history_window([asset], dt, -1, ['close'], 'daily')
            preclose = pre_value[asset.sid]['close'].iloc[-1]
            open_pct = spot_value['open'] / preclose - 1
        return open_pct, preclose

    def get_window(self,
                   assets,
                   dt,
                   days_in_window,
                   field,
                   data_frequency):
        """
        Internal method that gets a window of raw daily data for a sid
        and specified date range.  Used to support the history API method for
        daily bars.

        Parameters
        ----------
        assets : list --- element is Asset
            The asset whose data is desired.

        dt: pandas.Timestamp
            The end of the desired window of data.

        field: string or list
            The specific field to return.  "open", "high", "close_price", etc.

        days_in_window: int
            The number of days of data to return.

        data_frequency : minute or daily

        Returns
        -------
        A numpy array with requested values.  Any missing slots filled with
        nan.
        """
        history_reader = self._history_loader[data_frequency]
        window_array = history_reader.window(assets, field, dt, days_in_window)
        return window_array

    def get_history_window(self,
                           assets,
                           end_date,
                           bar_count,
                           field,
                           data_frequency):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ----------
        assets : list of zipline.data.Asset objects
            The asset whose data is desired.

        end_date : history date(not include)

        bar_count: int
            The number of bars desired.

        frequency: string
            "1d" or "1m"

        field: string or list
            The desired field of the asset.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars.

        ex: boolean
            raw or adjusted array

        Returns
        -------
        A dataframe containing the requested data.
        """
        fields = field if isinstance(field, (set, list)) else [field]
        if not set(field).issubset(self.OHLCV_FIELDS):
            raise ValueError("Invalid field: {0}".format(field))

        if abs(bar_count) < 1:
            raise ValueError(
                "abs (bar_count) must be >= 1, but got {}".format(bar_count)
            )
        history = self._history_loader[data_frequency]
        history_window_arrays = history.history(assets, fields, end_date, bar_count)
        return history_window_arrays

    def handle_extra_source(self):
        """
            extra data source
        """
        raise NotImplementedError()

    def sample_by_freq(self, freq, args, kwargs):
        """
            by_minute  :param kwargs: hour,minute
            by_week :param delta: int , the number day of week (1-7) which is trading_day
            by_month :param delta: int ,the number day of month (max -- 31) which is trading_day
        """
        method_name = '%s_rules' % freq
        samples = getattr(self.freq_rule, method_name)(args, kwargs)
        return samples

    @staticmethod
    def get_current_minutes(sid):
        """
            return current live tickers data
        """
        _url = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?fields1=f1&' \
               'fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0&secid={}'
        # 处理数据
        req_sid = '0.' + sid if sid.startswith('6') else '1.' + sid
        req_url = _url.format(req_sid)
        obj = _parse_url(req_url, bs=False)
        d = json.loads(obj)
        raw_array = [item.split(',') for item in d['data']['trends']]
        minutes = pd.DataFrame(raw_array, columns=['ticker', 'open', 'close', 'high',
                                                   'low', 'volume', 'turnover', 'avg'])
        return minutes

    @staticmethod
    def get_equities_pledge(symbol):
        frame = tsclient.to_ts_pledge(symbol)
        return frame


portal = DataPortal()

__all__ = ['portal']


# if __name__ == '__main__':
#
#     from gateway.asset.assets import Equity
#     data_portal = DataPortal()
#     asset = Equity('000595')
#     fields = ['volume']
#     sliding_window = portal.get_window([asset], '2019-09-02', - abs(5), ['volume'], 'daily')
#     print('sliding_window', sliding_window)
#     threshold = sliding_window[asset.sid]['volume'].mean() * 0.1
#     print('threshold', threshold)

# if __name__ == '__main__':
#
#     from gateway.asset.assets import Equity
#     data_portal = DataPortal()
#     assets = [Equity('600000')]
#     fields = ['open', 'close', 'amount']
#     # fields = ['open', 'close']
#     sessions = ['2020-05-01', '2020-09-30']
#     day_window = 26
#
#     divdends = data_portal.get_dividends(assets, '2017-05-25')
#     print('divdends', divdends)
#
#     rights = data_portal.get_rights(assets, '2000-01-24')
#     print('rights', rights)
#
#     open_pct, preclose = data_portal.get_open_pct(assets[0], '2020-09-03')
#     print('open_pct and preclose', open_pct, preclose)
#
#     daily_window_data = data_portal.get_window(assets, sessions[1], days_in_window=day_window,
#                                                field=fields, data_frequency='daily')
#     print('daily_window_data', daily_window_data)
#
#     minute_window_data = data_portal.get_window(assets, sessions[0], days_in_window=-day_window,
#                                                 field=fields, data_frequency='minute')
#     print('minute_window_data', minute_window_data)
#
#     history_daily_data = data_portal.get_history_window(assets, sessions[0],
#                                                         bar_count=-day_window, field=fields, data_frequency='daily')
#     print('history_daily_data', history_daily_data)
#
#     history_minute_data = data_portal.get_history_window(assets, sessions[0],
#                                                          bar_count=-300, field=fields, data_frequency='minute')
#     print('history_minute_data', history_minute_data)
#
#     daily_spot_value = data_portal.get_spot_value('2020-09-03', assets[0], 'daily', ['close', 'low'])
#     print('daily_spot_value', daily_spot_value)
#
#     minute_spot_value = data_portal.get_spot_value('2005-09-07', assets[0], 'minute', ['close'])
#     print('minute_spot_value', minute_spot_value, minute_spot_value.index[0])
#
#     daily_stack_value = data_portal.get_stack_value('equity', sessions[0], -day_window, 'daily')
#     print('daily_stack_value', daily_stack_value)
#
#     minute_stack_value = data_portal.get_stack_value('equity', sessions[0], -day_window, 'minute')
#     print('minute_stack_value', minute_stack_value)
