# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import defaultdict
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
        #
        if dts[0] < expires[0] or dts[0] > expires[-1]:
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
    def __init__(self, cache=None, cleanup=lambda value_to_clean: None):
        if cache is not None:
            self._cache = cache
        else:
            self._cache = {}

        self.cleanup = cleanup

    def get(self, key, dt):
        """Get the value of a cached object.

        Parameters
        ----------
        key : any
            The key to lookup.
        dt : datetime
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
        print('key', key)
        try:
            return self._cache[key].unwrap(dt)
        except Expired:
            self.cleanup(self._cache[key]._unsafe_get_value())
            del self._cache[key]
        except KeyError:
            pass

    def set(self, key, value, expiration_dt):
        """Adds a new key value pair to the cache.

        Parameters
        ----------
        key : any
            The key to use for the pair.
        value : any
            The value to store under the name ``key``.
        expiration_dt : datetime
            When should this mapping expire? The cache is considered invalid
            for dates **strictly greater** than ``expiration_dt``.
        """
        self._cache[key] = CachedObject(value, expiration_dt)


class HistoryLoader(ABC):

    _window_blocks = defaultdict(ExpiredCache)

    @property
    def trading_calendar(self):
        return calendar

    @property
    def frequency(self):
        raise NotImplementedError()

    @abstractmethod
    def _compute_slice_window(self, data, date, window):
        raise NotImplementedError

    def _ensure_sliding_windows(self, dts, assets, field):
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
        field : str or list
            The OHLCV field for which to retrieve data.
        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        asset_windows = {}
        needed_assets = []
        for asset in assets:
            try:
                _window = self._window_blocks[asset.sid].get(
                    field, dts)
            except KeyError or Expired:
                needed_assets.append(asset)
            else:
                slice_window = self._compute_slice_window(_window, dts)
                asset_windows[asset] = slice_window

        if needed_assets:
            for i, asset in enumerate(needed_assets):
                sliding_window = self.adjust_window.window_arrays(
                        dts,
                        asset,
                        field
                            )
                asset_windows[asset] = sliding_window
                # ExpiredCache
                self._window_blocks[asset].set(
                    field,
                    sliding_window,
                    dts)
        return [asset_windows[asset] for asset in assets]

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
        session = self.trading_calendar.session_in_window(dts, window, include=False)

        if len(session) == 1:
            block_arrays = self.adjust_window.array(session, assets, field)
        else:
            block_arrays = self._ensure_sliding_windows(
                                            session,
                                            assets,
                                            field
                                            )
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
        # 不同频率 --- 单独设立
        self._window_blocks = defaultdict(ExpiredCache)

    @property
    def frequency(self):
        return 'daily'

    def _compute_slice_window(self, _window, dts):
        sessions = calendar.session_in_range(*dts, include=True)
        slice_window = _window.reindex(sessions)
        return slice_window


class HistoryMinuteLoader(HistoryLoader):

    def __init__(self,
                 _minute_reader,
                 equity_adjustment_reader):
        self.adjust_window = AdjustedMinuteWindow(
                                            _minute_reader,
                                            equity_adjustment_reader)
        self._window_blocks = defaultdict(ExpiredCache)

    @property
    def frequency(self):
        return 'minute'

    def _compute_slice_window(self, _window, dts):
        ticker = np.clip(np.array(_window.index), *dts)
        _slice_window = _window.reindex(ticker)
        return _slice_window


if __name__ == '__main__':

    minute_reader = BcolzMinuteReader()
    session_reader = AssetSessionReader()
    adjustment_reader = SQLiteAdjustmentReader()

    asset = Equity('000001')
    sessions = ['2010-08-10', '2015-10-30']
    # minute_history = HistoryMinuteLoader(minute_reader, adjustment_reader)
    daily_history = HistoryDailyLoader(session_reader, adjustment_reader)
    his = daily_history.history([asset], ['close', 'open'], sessions[0], 10)
    print('his', his)
