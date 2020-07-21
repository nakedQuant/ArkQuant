# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from toolz import keyfilter, valmap
import pandas as pd,json
from .tools import  _parse_url
from .history_loader import (
    HistoryDailyLoader,
    HistoryMinuteLoader
)


class DataPortal(object):
    """Interface to all of the data that a simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of assets on a given day or to service history
    calls.

    Parameters
    ----------
    asset_finder : zipline.assets.assets.AssetFinder
        The AssetFinder instance used to resolve assets.
    trading_calendar: zipline.utils.calendar.exchange_calendar.TradingCalendar
        The calendar instance used to provide minute->session information.
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

    Asset_Type = frozenset(['symbol','etf','bond'])

    def __init__(self,
                asset_finder,
                trading_calendar,
                first_trading_day,
                _dispatch_session_reader,
                _dispatch_minute_reader,
                adjustment_reader,
                 ):
        self.asset_finder = asset_finder

        self.trading_calendar = trading_calendar

        self._first_trading_day = first_trading_day

        self._adjustment_reader = adjustment_reader

        self._pricing_readers = {
            'minute': _dispatch_minute_reader,
            'daily': _dispatch_session_reader,
        }

        _history_daily_loader = HistoryDailyLoader(
            _dispatch_minute_reader,
            self._adjustment_reader,
            trading_calendar,
        )
        _history_minute_loader = HistoryMinuteLoader(
            _dispatch_session_reader,
            self._adjustment_reader,
            trading_calendar,

        )
        self._history_loader = {
            'daily':_history_daily_loader,
            'minute':_history_minute_loader,
        }

        # Get the first trading minute
        self._first_trading_minute, _ = (
            self.trading_calendar.open_and_close_for_session(
                [self._first_trading_day]
            )
            if self._first_trading_day is not None else (None, None)
        )

        # Store the locs of the first day and first minute
        self._first_trading_day_loc = (
            self.trading_calendar.all_sessions.get_loc(self._first_trading_day)
            if self._first_trading_day is not None else None
        )
        self._extra_source = None

    @property
    def adjustment_reader(self):
        return self._adjustment_reader

    def _get_pricing_reader(self, data_frequency):
        return self._pricing_readers[data_frequency]

    def get_fetcher_assets(self, _typ):
        """
        Returns a list of assets for the current date, as defined by the
        fetcher data.

        Returns
        -------
        list: a list of Asset objects.
        """
        # return a list of assets for the current date, as defined by the
        # fetcher source
        assets = self.asset_finder.lookup_assets(_typ)
        return assets

    def get_dividends(self, sids, trading_days):
        """
        splits --- divdends

        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """
        extra = set(sids) - set(self._divdends_cache)
        if extra:
            for sid in extra:
                divdends = self.adjustment_reader.load_splits_for_sid(sid)
                self._divdends_cache[sid] = divdends
        #
        from toolz import keyfilter,valmap
        cache  = keyfilter(lambda x : x in sids,self._splits_cache)
        out = valmap(lambda x : x[x['pay_date'].isin(trading_days)] if x else x ,cache)
        return out

    def get_stock_rights(self, sids, trading_days):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """
        extra = set(sids) - set(self._rights_cache)
        if extra:
            for sid in extra:
                rights = self.adjustment_reader.load_splits_for_sid(sid)
                self._rights_cache[sid] = rights
        #
        cache  = keyfilter(lambda x : x in sids,self._rights_cache)
        out = valmap(lambda x : x[x['pay_date'].isin(trading_days)] if x else x ,cache)
        return out

    def _get_history_sliding_window(self,assets,
                                    end_dt,
                                    fields,
                                    bar_count,
                                    frequency
                                   ):
        """
        Internal method that returns a dataframe containing history bars
        of minute frequency for the given sids.
        """
        history = self._history_daily_loader[frequency]
        history_arrays = history.history(assets,fields,end_dt,window = bar_count)
        return history_arrays

    def get_history_window(self,
                           assets,
                           end_dt,
                           bar_count,
                           field,
                           data_frequency):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ----------
        assets : list of zipline.data.Asset objects
            The assets whose data is desired.

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
        history_window_arrays = self._get_history_sliding_window(assets,
                                                             end_dt,
                                                             field,
                                                             bar_count,
                                                             data_frequency)
        return history_window_arrays

    def get_window_data(self,
                         assets,
                         dt,
                         field,
                         days_in_window,
                         frequency):
        """
        Internal method that gets a window of adjusted daily data for a sid
        and specified date range.  Used to support the history API method for
        daily bars.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.

        dt: pandas.Timestamp
            The end of the desired window of data.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        bar_count: int
            The number of days of data to return.

        data_frequency : minute or daily

        Returns
        -------
        A numpy array with requested values.  Any missing slots filled with
        nan.
        """
        _reader = self._get_pricing_readers[frequency]
        window_array = _reader.load_raw_arrays(dt, days_in_window, field, assets)
        return window_array

    def _get_resized_minutes(self,dts,sids,field,_ticker):
        """
            Internal method that resample
            api : groups.keys() , get_group()
        """
        _minutes_reader = self._pricing_readers['minute']
        resamples = _minutes_reader.reindex_minutes_ticker(dts,sids,field,_ticker)
        return resamples

    def get_resample_minutes(self,sessions,sids,field,frequency):
        reindex_minutes = self._get_resized_minutes(sessions,sids,field,frequency)
        return reindex_minutes

    def get_current(self,sid):
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
        minutes = pd.DataFrame(raw_array,
                          columns=['ticker', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'avg'])
        return minutes

    def handle_extra_source(self,source_df):
        """
            Internal method that determines if this asset/field combination
            represents a fetcher value or a regular OHLCVP lookup.
            Extra sources always have a sid column.
            We expand the given data (by forward filling) to the full range of
            the simulation dates, so that lookup is fast during simulation.
        """
        raise NotImplementedError()