# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, json
from functools import lru_cache
from .tools import _parse_url
from .third_api.client import tsclient
from .history_loader import (
    HistoryDailyLoader,
    HistoryMinuteLoader
)


class DataPortal(object):
    """Interface to all of the data that a simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of asset on a given day or to service history
    calls.

    Parameters
    ----------
    asset_finder : zipline.assets.assets.AssetFinder
        The AssetFinder instance used to resolve asset.
    trading_calendar: zipline.utils._calendar.exchange_calendar.TradingCalendar
        The _calendar instance used to provide minute->session information.
    first_trading_day : pd.Timestamp
        The first trading day for the simulation.
    equity_daily_reader : BcolzDailyBarReader, optional
        The daily bar reader for equities. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    equity_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for equities. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    adjustment_reader : SQLiteAdjustmentWriter, optional
        The adjustment reader. This is used to apply splits, dividends, and
        other adjustment data to the raw data from the readers.
    """

    OHLCV_FIELDS = frozenset(["open", "high", "low", "close", "volume"])

    def __init__(self,
                 asset_finder,
                 _session_reader,
                 _minute_reader,
                 adjustment_reader):

        self.asset_finder = asset_finder

        self._adjustment_reader = adjustment_reader

        self._pricing_reader = {
            'minute': _minute_reader,
            'daily': _session_reader,
        }

        _history_daily_loader = HistoryDailyLoader(
            _minute_reader,
            self._adjustment_reader,
        )
        _history_minute_loader = HistoryMinuteLoader(
            _session_reader,
            self._adjustment_reader,

        )
        self._history_loader = {
            'daily': _history_daily_loader,
            'minute': _history_minute_loader,
        }
        self._extra_source = None

    @property
    def adjustment_reader(self):
        return self._adjustment_reader

    def get_fetcher_assets(self, sids):
        """
        Returns a list of asset for the current date, as defined by the
        fetcher data.

        Returns
        -------
        list: a list of Asset objects.
        """
        # return a list of asset for the current date, as defined by the
        # fetcher source
        found, missing = self.asset_finder.retrieve_asset(sids)
        return found, missing

    def get_all_assets(self, asset_type=None):
        all_assets = self.asset_finder.retrieve_all(asset_type)
        return all_assets

    def get_dividends_for_sid(self, sid, trading_day):
        """
        splits --- divdends

        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_day: pd.DatetimeIndex
            The trading day.

        Returns
        -------
            equity divdends or cash divdends
        """
        divdends = self._adjustment_reader.load_divdend_for_sid(sid, trading_day)
        return divdends

    def get_rights_for_sid(self, sid, trading_day):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_day: pd.DatetimeIndex
            The trading dt.

        Returns
        -------
            equity rights
        """
        rights = self._adjustment_reader.load_right_for_sid(sid, trading_day)
        return rights

    @lru_cache(maxsize=32)
    def _retrieve_pct(self, dts):
        pct = self._pricing_reader['daily'].get_stock_pct(dts)
        return pct

    def get_open_pct(self, asset, dts):
        # 获取标的pct_change
        frame = self._retrieve_pct(dts)
        pct = frame.loc[asset.sid, 'pct']
        # 获取close
        kline = self.get_spot_value(asset, dts, 'daily', ['open', 'close'])
        # 计算close
        preclose = kline['close'] / (1 + pct)
        open_pct = kline['open'][-1] / preclose
        return open_pct, preclose

    def get_spot_value(self, asset, dts, frequency, fields):
        spot_value = self._pricing_reader[frequency].get_spot_value(dts, asset, fields)
        return spot_value

    def _get_history_sliding_window(self,
                                    assets,
                                    end_dt,
                                    fields,
                                    bar_count,
                                    frequency):
        """
            Internal method that gets a window of adjusted daily data for a sid
            and specified date range.  Used to support the history API method for
            daily bars.
        """
        history = self._history_loader[frequency]
        history_arrays = history.history(assets, fields, end_dt, bar_count)
        return history_arrays

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

        field: string
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
        if field not in self.OHLCV_FIELDS:
            raise ValueError("Invalid field: {0}".format(field))

        if bar_count < 1:
            raise ValueError(
                "bar_count must be >= 1, but got {}".format(bar_count)
            )
        history_window_arrays = self._get_history_sliding_window(
                                                            assets,
                                                            end_date,
                                                            field,
                                                            bar_count,
                                                            data_frequency)
        return history_window_arrays

    def get_window_data(self,
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
        _reader = self._pricing_reader[data_frequency]
        sessions = self.trading_calendar.session_in_window(dt, days_in_window, False)
        window_array = _reader.load_raw_arrays(sessions, assets, field)
        return window_array

    def get_resize_data(self,dt,window,freq,assets,field):
        """
            return resample daily kline --- Year Month Day
        """
        resample_data = self._history_loader['daily'].get_resampled(
                                                                dt,
                                                                window,
                                                                freq,
                                                                assets,
                                                                field
                                                                    )
        return resample_data

    def get_specific_ticker_data(self, dt, window, ticker, assets, field):
        """
            eg --- 9:30 or 11:20
        """
        resample_tickers = self._history_loader['minute'].get_resampled(
                                                                dt,
                                                                window,
                                                                ticker,
                                                                assets,
                                                                field
                                                                    )
        return resample_tickers

    @staticmethod
    def get_current(sid):
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

    def handle_extra_source(self, source_df):
        """
            Internal method that determines if this asset/field combination
            represents a fetcher value or a regular OHLCVP lookup.
            Extra sources always have a sid column.
            We expand the given data (by forward filling) to the full range of
            the simulation dates, so that lookup is fast during simulation.
        """
        raise NotImplementedError()

    @staticmethod
    def get_equities_pledge(symbol):
        frame = tsclient.to_ts_pledge(symbol)
        return frame

    @staticmethod
    def get_equity_adjfactor(code):
        factor = tsclient.to_ts_adjfactor(code)
        return factor
