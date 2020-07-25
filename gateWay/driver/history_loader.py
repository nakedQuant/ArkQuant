# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from collections import defaultdict
from abc import ABC , abstractmethod
import pandas as pd
from gateWay.driver.adjustArray import (
    AdjustedDailyWindow,
    AdjustedMinuteWindow
)

Seconds_Per_Day = 24 * 60 * 60


class Expired(Exception):
    """
        mark a cacheobject has expired
    """

#cache value dt
class CachedObject(object):
    """
    A simple struct for maintaining a cached object with an expiration date.

    Parameters
    ----------
    value : object
        The object to cache.
    expires : datetime-like
        Expiration date of `value`. The cache is considered invalid for dates
        **strictly greater** than `expires`.
    """
    def __init__(self, value, expires):
        self._value = value
        self._expires = expires

    def unwrap(self, dts):
        """
        Get the cached value.

        Returns
        -------
        value : object
            The cached value.

        Raises
        ------
        Expired
            Raised when `dt` is greater than self.expires.
        """
        # expires = self._expires
        # if expires is AlwaysExpired or expires < dt:
        #     raise Expired(self._expires)
        expires = self._expires
        if not set(dts).issubset(set(expires)):
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
        try:
            return self._cache[key].unwrap(dt)
        except Expired:
            self.cleanup(self._cache[key]._unsafe_get_value())
            del self._cache[key]
            raise KeyError(key)

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

    @property
    def frequency(self):
        raise NotImplementedError()

    @abstractmethod
    def _compute_slice_window(self,data,dt,window):
        raise NotImplementedError

    def _ensure_adjust_windows(self, edate, window,assets,field):
        """
        Ensure that there is a Float64Multiply window for each asset that can
        provide data for the given parameters.
        If the corresponding window for the (assets, len(dts), field) does not
        exist, then create a new one.
        If a corresponding window does exist for (assets, len(dts), field), but
        can not provide data for the current dts range, then create a new
        one and replace the expired window.

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str or list
            The OHLCV field for which to retrieve data.
        is_perspective_after : bool
            see: `PricingHistoryLoader.history`

        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        dts = self._trading_calendar.sessions_in_range(edate,window)
        #设立参数
        asset_windows = {}
        needed_assets = []
        #默认获取OHLCV数据
        for asset in assets:
            try:
                _window = self._window_blocks[asset].get(
                    field, dts)
            except KeyError:
                needed_assets.append(asset)
            else:
                _slice = self._compute_slice_window(_window,dts)
                asset_windows[asset] = _slice

        if needed_assets:
            for i, asset in enumerate(needed_assets):
                sliding_window = self.adjust_window._window_arrays(
                        edate,
                        window,
                        asset,
                        field
                            )
                asset_windows[asset] = sliding_window
                #设置ExpiredCache
                self._window_blocks[asset].set(
                    field,
                    sliding_window)
        return [asset_windows[asset] for asset in assets]

    @abstractmethod
    def get_resampled(self,dts,window,freq,assets,field):
        raise NotImplementedError()

    def history(self,dts,window,assets,field):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current simulation time.
        default fields --- OHLCV

        Parameters
        ----------
        assets : iterable of Assets
            The assets in the window.
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str or list
            The OHLCV field for which to retrieve data.
        window : int
            The length of window
        Returns
        -------
        out : np.ndarray with shape(len(days between start, end), len(assets))
        """
        if window != 0:
            block_arrays = self._ensure_sliding_windows(
                                            dts,
                                            window,
                                            assets,
                                            field
                                            )
        else:
            block_arrays = self.adjust_window._array([dts,dts],assets,field)
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
        self._trading_calendar = _daily_reader.calendar
        self._window_blocks = defaultdict(ExpiredCache())

    @property
    def frequency(self):
        return 'daily'

    def _compute_slice_window(self,_window,sessions):
        _slice_window = _window.reindex(sessions)
        return _slice_window

    @abstractmethod
    def get_resampled(self,dts,window,freq,assets,field):
        """
            select specific dts  Year Month Day
        """
        resampled = {}
        his = self.history(dts,window,assets,field)
        sdate = self._trading_calendar._roll_forward(dts,window)
        pds = [dt.strftime('%Y%m%d') for dt in pd.date_range(sdate,dts,freq = freq)]
        for sid,raw in his.items():
            resampled[sid] = raw.reindex(pds)
        return resampled


class HistoryMinuteLoader(HistoryLoader):

    def __init__(self,
                _minute_reader,
                 equity_adjustment_reader):
        self.adjust_minute_window = AdjustedMinuteWindow(
                                            _minute_reader,
                                            equity_adjustment_reader)
        self._trading_calendar = _minute_reader.calendar
        self._cache = {}

    @property
    def frequency(self):
        return 'minute'

    def _compute_slice_window(self,raw,dts):
        # 时间区间为子集，需要过滤
        dts_minutes = self._trading_calendar.minutes_in_window(dts)
        _slice_window = raw.reindex(dts_minutes)
        return _slice_window

    @abstractmethod
    def get_resampled(self,dts,window,freq,assets,field):
        """
            select specific dts minutes ,e,g --- 9:30,10:30
        """
        resamples = {}
        his = self.history(dts,window,assets,field)
        for sid,raw in his.items():
            seconds = dts.split(':')[0] * 60 * 60 + dts.split(':')[0] * 60
            ticker_index = map(lambda x : (x - seconds) / Seconds_Per_Day == 0 , raw.index)
            resamples[sid] = raw.reindex(ticker_index)
        return resamples


__all__ = [HistoryMinuteLoader,HistoryDailyLoader]