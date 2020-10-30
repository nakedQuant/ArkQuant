# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from _calendar.trading_calendar import calendar
from gateway.driver.adjustArray import (
                        AdjustedDailyWindow,
                        AdjustedMinuteWindow)

DefaultFields = frozenset(['open', 'high', 'low', 'close', 'amount', 'volume'])


class HistoryLoader(object):

    @property
    def trading_calendar(self):
        return calendar

    @property
    def frequency(self):
        raise NotImplementedError()

    def get_spot_value(self, dt, asset, fields):
        spot = self.adjust_window.get_spot_value(dt, asset, fields)
        return spot

    def get_stack_value(self, tbl, dt, window):
        sdate = self.trading_calendar.dt_window_size(dt, window)
        stack = self.adjust_window.get_stack_value(tbl, [sdate, dt])
        return stack

    def _ensure_sliding_windows(self, assets, fields, dts, window):
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
        sdate = self.trading_calendar.dt_window_size(dts, window)
        session = [sdate, dts]
        adjust_arrays = self.adjust_window.window_arrays(
            session,
            assets,
            list(DefaultFields)
        )
        from toolz import valmap
        sliding_window = valmap(lambda x: x.reindex(columns=fields), adjust_arrays)
        return sliding_window

    def window(self, assets, field, dts, window):
        if window == -1:
            frame = dict()
            date = self.trading_calendar.dt_window_size(dts, -1)
            for asset in assets:
                frame[asset.sid] = self.get_spot_value(date, asset, field)
        else:
            sessions = self.trading_calendar.session_in_window(dts, window)
            frame = self.adjust_window.array([min(sessions), max(sessions)], assets, field)
        return frame

    def history(self, assets, field, dts, window):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current ArkQuant time.
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
            block_arrays = self._ensure_sliding_windows(
                                            assets,
                                            field,
                                            dts,
                                            window
                                            )
        else:
            block_arrays = self.window(assets, field, dts, window)
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

    @property
    def frequency(self):
        return 'daily'

    def get_mkv_value(self, session, assets, fields):
        mkv = self.adjust_window.get_mkv_value(session, assets, fields)
        return mkv

    def window(self, assets, field, dts, window):
        daily_window = super().window(assets, field, dts, window)
        return daily_window

    def history(self, assets, field, dts, window):
        history_daily_arrays = super().history(assets, field, dts, window)
        return history_daily_arrays


class HistoryMinuteLoader(HistoryLoader):

    def __init__(self,
                 _minute_reader,
                 equity_adjustment_reader):
        self.adjust_window = AdjustedMinuteWindow(
                                            _minute_reader,
                                            equity_adjustment_reader)

    @property
    def frequency(self):
        return 'minute'

    def window(self, assets, field, dts, window):
        minutes_window = super().window(assets, field, dts, window)
        return minutes_window

    def history(self, assets, field, dts, window):
        history_minutes_arrays = super().history(assets, field, dts, window)
        return history_minutes_arrays


__all__ = [
    'HistoryMinuteLoader',
    'HistoryDailyLoader'
]


# if __name__ == '__main__':
#     from gateway.asset.assets import Equity, Convertible, Fund
#     from gateway.driver.bar_reader import AssetSessionReader
#     from gateway.driver.bcolz_reader import BcolzMinuteReader
#     from gateway.driver.adjustment_reader import SQLiteAdjustmentReader
#
#     minute_reader = BcolzMinuteReader()
#     session_reader = AssetSessionReader()
#     adjustment_reader = SQLiteAdjustmentReader()
#
#     # asset = Equity('600000')
#     asset = {Equity('000702'), Equity('000717'), Equity('000718'), Equity('000701'), Equity('000728'),
#              Equity('000712'), Equity('000713'), Equity('000689'), Equity('000727'), Equity('000721')}
#     sessions = ['2019-08-03', '2019-09-03']
#     fields = ['open', 'close']
#     daily_history = HistoryDailyLoader(session_reader, adjustment_reader)
#     his_window_daily = daily_history.history(asset, fields, sessions[1], window=-26)
#     print('sid', his_window_daily)
#     window_daily = daily_history.window(asset, fields, sessions[1], window=-26)
#     print('window_daily', window_daily)
