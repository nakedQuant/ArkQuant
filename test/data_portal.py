# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
from functools import lru_cache

from trading_calendar import Calendar
from adjustments import SQLiteAdjustmentReader
from reader import BarReader
from gateWay.assets.assets.asset import  Asset,Equity,Convertible,Fund
from gateWay.assets.asset_ext import  AssetFinder
from db_schema import ENGINE_PTH

BASE_FIELDS = frozenset([
    "open",
    "high",
    "low",
    "close",
    "turnover",
    "volume",
    "amount",
])

EXTRA_FIELDS = frozenset([
    "GDP",
    "Margin",
    "Massive",
    "Release",
    "shareHolder",
    "",
    "Exchange",
])


class DataPortal(object):
    """Interface to all of the data that a zipline simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of assets on a given day or to service history
    calls.
    """
    def __init__(self):

        self._reader = {
            'pricing':BarReader(ENGINE_PTH),
            'adjustment':SQLiteAdjustmentReader(ENGINE_PTH),
        }
        self.asset_finder = AssetFinder(ENGINE_PTH)
        self.trading_calendar = Calendar()

    @property
    def adjustment_reader(self):
        return self._reader['adjustment']

    @staticmethod
    def _is_extra_source(asset, field):
        """
        Internal method that determines if this asset/field combination
        represents a fetcher value or a regular OHLCVP lookup.
        """
        # If we have an extra source with a column called "price", only look
        # at it if it's on something like palladium and not AAPL (since our
        # own price data always wins when dealing with assets).

        return not (field in BASE_FIELDS and
                    (isinstance(asset, Asset)))

    @lru_cache(maxsize=32)
    def load_pricing_adjustment(self, sid,date):
        """
        Internal method that returns a list of adjustments for the given sid.
        Parameters
        ----------
        assets : container
            assets for which we want splits.
        dt : pd.Timestamp
            The date for which we are checking for splits. Note: this is
            expected to be midnight UTC.
        """
        divdends = self.adjustment_reader.\
            retrieve_divdend_info(sid,date)
        return divdends

        return adjustments

    def get_stock_rights(self,sid,date):
        """
            stock rights --- 配股
        """
        result = self._load_pricing_adjustment(sid,date)
        return result['rights']

    def get_stock_splits_dividends(self, sid, date):
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
        result = self._load_pricing_adjustment(sid,date)
        return result['splits_divdend']

    def _calculate_adjustments_ratio(self, asset,dt):
        """
        Returns a list of adjustments between the dt and perspective_dt for the
        given field and list of assets

        Parameters
        ----------
        assets : list of type Asset, or Asset
            The asset, or assets whose adjustments are desired.
        field : {'open', 'high', 'low', 'close', 'volume', \
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.

        Returns
        -------
        adjustments : list[Adjustment]
            The adjustments to that field.
        """
        adjustment_ratios_qfq = self.adjustment_reader.\
            load_adjustment_coef_for_sid(asset,dt)

        return adjustment_ratios_qfq

    def get_adjusted_value(self, asset, fields,sessions):
        """
        Returns a scalar value representing the value
        of the desired asset's field at the given dt with adjustments applied.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.
        field : {'open', 'high', 'low', 'close', 'volume', \
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.
        Returns
        -------
        value : float, int, or pd.Timestamp
            The value of the given ``field`` for ``asset`` at ``dt`` with any
            adjustments known by ``perspective_dt`` applied. The return type is
            based on the ``field`` requested. If the field is one of 'open',
            'high', 'low', 'close', or 'price', the value will be a float. If
            the ``field`` is 'volume' the value will be a int. If the ``field``
            is 'last_traded' the value will be a Timestamp.
        """
        if set(fields) not in BASE_FIELDS:
            raise KeyError("Invalid column: " + str(fields))
        try:
            sdate, edate = sessions
        except :
            print('cannot unpack sessions tuple')
        spot_value,should = self.get_spot_value(asset, fields,
                                         sdate,edate)
        trading_days = self.trading_calendar.session_in_range(sdate,edate)
        if should:
            ratio = self._calculate_adjustments_ratio(asset,edate)
            ratio = ratio.reindex(trading_days)
            ratio.fillna(method = 'bfill',inplace=True)
            ratio.fillna(method = 'ffill',inplace=True)
        else:
            ratio = pd.Series(1.0,index = trading_days)
        # adjusted_value = self.map_adjust(spot_value,ratio)
        return spot_value,ratio

    def get_spot_value(self, asset,fields,sdate,edate):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ----------
        assets : Asset, ContinuousFuture, or iterable of same.
            The asset or assets whose data is desired.
        field : {'open', 'high', 'low', 'close', 'volume'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The spot value of ``field`` for ``asset`` The return type is based
            on the ``field`` requested. If the field is one of 'open', 'high',
            'low', 'close', or 'price', the value will be a float. If the
            ``field`` is 'volume' the value will be a int. If the ``field`` is
            'last_traded' the value will be a Timestamp.
        """
        pricing_reader = self._reader['pricing']
        should_adjust = False
        if isinstance(asset,Equity):
            df = pricing_reader.load_daily_symbol(
                sdate,edate,fields,asset)
            should_adjust = True
        elif isinstance(asset,Convertible):
            df = pricing_reader.load_daily_bond(
                sdate,edate,fields,asset)
        elif isinstance(asset,Fund):
            df = pricing_reader.load_daily_fund(
                sdate,edate,fields,asset)
        else:
            raise ValueError('unkown asset type %s'%asset)
        return df,should_adjust

    def get_day_minutes_from_now(self, asset,lag = 5):
        """
        Internal method that returns a dataframe containing history bars
        of minute frequency for the given sids.
        """
        lag = 5 if lag > 5 else lag
        minutes_df = self._pricing_readers.load_minutes(asset,lag)
        return minutes_df