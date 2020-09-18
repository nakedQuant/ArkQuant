# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
import numpy as np
from _calendar.trading_calendar import calendar
from gateway.driver.bar_reader import AssetSessionReader
from gateway.driver.bcolz_reader import BcolzMinuteReader
from gateway.driver.adjustment_reader import SQLiteAdjustmentReader
from gateway.driver.adjustArray import (
                        AdjustedDailyWindow,
                        AdjustedMinuteWindow
                                        )
from gateway.asset.assets import Equity, Convertible, Fund


__all__ = [
    'HistoryMinuteLoader',
    'HistoryDailyLoader'
]

DefaultFields = frozenset(['open', 'high', 'low', 'close', 'amount', 'volume'])


class Expired(Exception):
    """
        mark a cacheobject has expired
    """


class CachedObject(object):
    """
    A simple struct for maintaining a cached object with an expiration date.

    Parameters
    ----------
    value : object
        The object to cache.
    expires : datetime-like []
        Expiration date of `value`. The cache is considered invalid for dates
        **strictly greater** than `expires`.
    """
    def __init__(self, value, expires):
        self._value = value
        self._expires = expires

    def unwrap(self, dts):
        """
        Get the cached value.
        dts: sessions
        dts : [start_date, end_date]

        Returns
        -------
        value : object
            The cached value.

        Raises
        ------
        Expired
            Raised when `dt` is greater than self.expires.
        """
        expires = self._expires
        if dts[0] < expires[0] or dts[-1] > expires[-1]:
            raise Expired(self._expired)
        return self._value

    def _unsafe_get_value(self):
        """You almost certainly shouldn't use this."""
        return self._value


class ExpiredCache(object):
    """
    A cache of multiple CachedObjects, which returns the wrapped the value
    or raises and deletes the CachedObject if the value has expired.

    Parameters
    ----------
    cache : dict-like, optional
        An instance of a dict-like object which needs to support at least:
        `__del__`, `__getitem__`, `__setitem__`
        If `None`, than a dict is used as a default.

    cleanup : callable, optional
        A method that takes a single argument, a cached object, and is called
        upon expiry of the cached object, prior to deleting the object. If not
        provided, defaults to a no-op.

    """
    def __init__(self):
        self._cache = {}
        # cleanup = lambda value_to_clean: None

    def get(self, key, dts):
        """Get the value of a cached object.

        Parameters
        ----------
        key : any
            The key to lookup.
        dts : datetime list e.g.[start, end]
            The time of the lookup.

        Returns
        -------
        result : any
            The value for ``key``.

        Raises
        ------
        KeyError
            Raised if the key is not in the cache or the value for the key
            has expired.
        """
        value = self._cache[key].unwrap(dts)
        return value

    def set(self, key, value, expiration_dt):
        """Adds a new key value pair to the cache.

        Parameters
        ----------
        key : sid
            Asset object sid attribute
        value : any
            The value to store under the name ``key``.
        expiration_dt : datetime
            When should this mapping expire? The cache is considered invalid
            for dates **strictly greater** than ``expiration_dt``.
        """
        self._cache[key] = CachedObject(value, expiration_dt)


class HistoryLoader(ABC):

    @property
    def trading_calendar(self):
        return calendar

    @property
    def frequency(self):
        raise NotImplementedError()

    def get_spot_value(self, dt, asset, fields):
        spot = self.adjust_window.get_spot_value(dt, asset, fields)
        return spot

    def get_stack_value(self, tbl, session):
        stack = self.adjust_window.get_stack_value(tbl, session)
        return stack

    @abstractmethod
    def _compute_slice_window(self, data, date, window):
        raise NotImplementedError

    def _ensure_sliding_windows(self, dts, assets, fields):
        """
        Ensure that there is a Float64Multiply window for each asset that can
        provide data for the given parameters.
        If the corresponding window for the (asset, len(dts), field) does not
        exist, then create a new one.
        If a corresponding window does exist for (asset, len(dts), field), but
        can not provide data for the current dts range, then create a new
        one and replace the expired window.

        Parameters
        ----------
        assets : iterable of Assets
            The asset in the window
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the _calendar.
        fields : str or list
            The OHLCV field for which to retrieve data.
        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        asset_windows = {}
        needed_assets = []
        for asset_obj in assets:
            # print('blocks', self._window_blocks)
            # print('asset', asset_obj)
            try:
                cache_window = self._window_blocks.get(
                    asset_obj, dts)
                print('cache_window', cache_window)
            except Expired:
                del self._window_blocks[asset_obj]
            except KeyError:
                needed_assets.append(asset_obj)
            else:
                slice_window = self._compute_slice_window(cache_window, dts)
                asset_windows[asset_obj.sid] = slice_window.loc[:, fields]

        if needed_assets:
            for i, target_asset in enumerate(needed_assets):
                sliding_window = self.adjust_window.window_arrays(
                        dts,
                        [target_asset],
                        list(DefaultFields)
                            )[target_asset.sid]
                # ExpiredCache
                self._window_blocks.set(
                    target_asset,
                    sliding_window,
                    dts)
                asset_windows[target_asset.sid] = sliding_window.loc[:, fields] \
                    if not sliding_window.empty else sliding_window
        return asset_windows

    def window(self, assets, field, dts, window):
        assert window < 0, 'to avoid forward prospective error'
        sessions = self.trading_calendar.session_in_window(dts, window)
        frame = self.adjust_window.array([min(sessions), max(sessions)], assets, field)
        return frame

    def history(self, assets, field, dts, window):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current nakedquant time.
        default fields --- OHLCV

        Parameters
        ----------
        assets : iterable of Assets
            The asset in the window.
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the _calendar.
        field : str or list
            The OHLCV field for which to retrieve data.
        window : int
            The length of window
        Returns
        -------
        out : np.ndarray with shape(len(days between start, end), len(asset))
        """
        # 不包括当天数据
        assert window < 0, 'to avoid forward prospective error'
        if window != -1:
            pre_dt = self.trading_calendar.dt_window_size(dts, window)
            # print('pre_dt', pre_dt)
            block_arrays = self._ensure_sliding_windows(
                                            [pre_dt, dts],
                                            assets,
                                            field
                                            )
        else:
             block_arrays = self.window(assets, field, dts, window=-1)
        return block_arrays


class HistoryDailyLoader(HistoryLoader):
    """
        生成调整后的序列
        优化 --- 缓存
    """

    def __init__(self,
                 _daily_reader,
                 equity_adjustment_reader):
        self.adjust_window = AdjustedDailyWindow(
                                            _daily_reader,
                                            equity_adjustment_reader)
        self._window_blocks = ExpiredCache()

    @property
    def frequency(self):
        return 'daily'

    @staticmethod
    def _compute_slice_window(_window, dts):
        print('_window', _window)
        # print('dts', dts)
        sessions = calendar.session_in_range(*dts)
        # print('sessions', sessions)
        slice_window = _window.reindex(sessions)
        return slice_window


class HistoryMinuteLoader(HistoryLoader):

    def __init__(self,
                 _minute_reader,
                 equity_adjustment_reader):
        self.adjust_window = AdjustedMinuteWindow(
                                            _minute_reader,
                                            equity_adjustment_reader)
        self._window_blocks = ExpiredCache()

    @property
    def frequency(self):
        return 'minute'

    @staticmethod
    def _compute_slice_window(_window, dts):
        ticker = np.clip(np.array(_window.index), *dts)
        _slice_window = _window.reindex(ticker)
        return _slice_window


if __name__ == '__main__':

    minute_reader = BcolzMinuteReader()
    session_reader = AssetSessionReader()
    adjustment_reader = SQLiteAdjustmentReader()

    asset = Equity('600000')
    sessions = ['2005-01-01', '2010-10-30']
    fields = ['open', 'close']
    # daily_history = HistoryDailyLoader(session_reader, adjustment_reader)
    # his_window_daily = daily_history.history([asset], fields, sessions[0], window=-30)
    # print('history_window_daily', his_window_daily)
    # his_daily = daily_history.history([asset], fields, '2010-12-31')
    # print('his', his_daily)
    # daily_spot_value = daily_history.get_spot_value('2005-09-07', asset, fields)
    # print('daily_spot_value', daily_spot_value)
    # daily_spot_value = daily_history.get_stack_value('equity', sessions)
    # print('daily_spot_value', daily_spot_value)
    # daily_open_pct = daily_history.get_open_pct([asset], '2005-09-07')
    # print('daily_open_pct', daily_open_pct)

    minute_history = HistoryMinuteLoader(minute_reader, adjustment_reader)
    # his_window_minute = minute_history.history([asset], ['close', 'open'], '2005-09-03', window=-1000)
    # print('his_window_minute', his_window_minute)
    his_minute = minute_history.history([asset], ['close', 'open'], '2005-09-08', -1)
    print('his_minute', his_minute)
    # minute_window = his_window_data = minute_history.window([asset], ['close', 'open'], '2005-09-07', -10)
    # print('window_data', minute_window)
    # minute_spot_value = minute_history.get_spot_value('2005-09-07', asset, fields)
    # print('minute_spot_value', minute_spot_value)
    # minute_stack_value = minute_history.get_stack_value('equity', sessions)
    # print('minute_stack_value', minute_stack_value)
