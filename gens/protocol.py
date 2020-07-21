# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

class BarData:
    """
    Provides methods for accessing minutely and daily price/volume data from
    Algorithm API functions.

    Also provides utility methods to determine if an asset is alive, and if it
    has recent trade data.

    An instance of this object is passed as ``data`` to
    :func:`~zipline.api.handle_data` and
    :func:`~zipline.api.before_trading_start`.

    Parameters
    ----------
    data_portal : DataPortal
        Provider for bar pricing data.
    data_frequency : {'minute', 'daily'}
        The frequency of the bar data; i.e. whether the data is
        daily or minute bars
    restrictions : zipline.finance.asset_restrictions.Restrictions
        Object that combines and returns restricted list information from
        multiple sources
    """

    def __init__(self, data_portal, data_frequency,
                 trading_calendar, restrictions):
        self.data_portal = data_portal
        self.data_frequency = data_frequency
        self._trading_calendar = trading_calendar
        self._is_restricted = restrictions.is_restricted

    def get_current_ticker(self,assets,fields):
        """
        Returns the "current" value of the given fields for the given assets
        at the current simulation time.
        :param assets: asset_type
        :param fields: OHLCTV
        :return: dict asset -> ticker
        intended to return current ticker
        """
        cur = {}
        for asset in assets:
            ticker = self.data_portal.get_current(asset)
            cur[asset] = ticker.loc[:,fields]
        return cur

    def history(self, assets, end_dt,bar_count, fields,frequency):
        """
        Returns a trailing window of length ``bar_count`` containing data for
        the given assets, fields, and frequency.

        Returned data is adjusted for splits, dividends, and mergers as of the
        current simulation time.

        The semantics for missing data are identical to the ones described in
        the notes for :meth:`current`.

        Parameters
        ----------
        assets: zipline.assets.Asset or iterable of zipline.assets.Asset
            The asset(s) for which data is requested.
        fields: string or iterable of string.
            Requested data field(s). Valid field names are: "price",
            "last_traded", "open", "high", "low", "close", and "volume".
        bar_count: int
            Number of data observations requested.
        frequency: str
            String indicating whether to load daily or minutely data
            observations. Pass '1m' for minutely data, '1d' for daily data.

        Returns
        -------
        history : pd.Series or pd.DataFrame or pd.Panel
            See notes below.

        Notes
        ------
        returned panel has:
        items: fields
        major axis: dt
        minor axis: assets
        return pd.Panel(df_dict)
        """
        sliding_window = self.data_portal.get_history_window(assets,
                                                             end_dt,
                                                             bar_count,
                                                             fields,
                                                             frequency)
        return sliding_window

    def window_data(self,assets,end_dt,bar_count,fields,frequency):
        window_array = self.data_portal.get_window_data(assets,
                                                        end_dt,
                                                        bar_count,
                                                        fields,
                                                        frequency)
        return window_array
