# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, bcolz, os
from .bar_reader import BarReader
from gateway.driver import TdxDir
from .tools import transfer_to_timestamp

__all__ = [
    'BcolzDailyReader',
    'BcolzMinuteReader'
]


class BcolzReader(BarReader):
    """
       restricted to equity
    """
    default = frozenset(['open', 'high', 'low', 'close', 'amount', 'volume'])

    def get_sid_attr(self, sid):
        sid_path = '{}.bcolz'.format(sid)
        root = os.path.join(self._root_dir, sid_path)
        return root

    def _read_bcolz_data(self, sid):
        """cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)"""
        root_dir = self.get_sid_attr(sid)
        table = bcolz.open(rootdir=root_dir, mode='r')
        return table

    def get_value(self, sid, sdate, edate):
        """
        Retrieve the pricing info for the given sid, dt, and field.

        Parameters
        ----------
        sid : int
            Asset identifier.
        sdate : datetime-like
            The datetime at which the trade occurred.
        edate : datetime-like
            The datetime at which the trade occurred.
        field : string
            The type of pricing data to retrieve.
            ('open', 'high', 'low', 'close', 'volume')

        Returns
        -------
        out : float|int

        The market data for the given sid, dt, and field coordinates.

        For OHLC:
            Returns a float if a trade occurred at the given dt.
            If no trade occurred, a np.nan is returned.

        For volume:
            Returns the integer value of the volume.
            (A volume of 0 signifies no trades for the given dt.)
        """
        table = self._read_bcolz_data(sid)
        meta = table.attrs['metadata']
        assert meta.end_session >= sdate, ('%r exceed metadata end_session'%sdate)
        #获取数据
        if self.data_frequency == 'minute':
            start = transfer_to_timestamp(sdate)
            end = transfer_to_timestamp(edate)
            condition = '({0} <= timestamp) & (timestamp <= {1)'.format(start, end)
        else:
            condition = '({0} <= trade_dt) & (trade_dt <= {1)'.format(sdate, edate)
        # raw = table.fetchwhere(condition)
        # frame = pd.DataFrame(raw)
        frame = pd.DataFrame(table.fetchwhere(condition))
        #调整系数
        inverse_ratio = 1 / meta.ohlc_ratio
        scale = frame.apply(lambda x: [x[col] * inverse_ratio for col in ['open', 'high', 'low', 'close']], axis=1)
        scale.set_index('ticker', inplace=True) if 'ticker' in scale.columns \
            else scale.set_index('trade_dt', inplace=True)
        return scale

    def get_spot_value(self, asset, dt, fields):
        raise NotImplementedError()

    def load_raw_arrays(self, sessions, assets, columns):
        assert set(columns) in self.default, 'unknown field'
        sdate, edate = sessions
        supplements = columns + ['trade_dt'] \
            if self.data_frequency == 'daily' else columns + ['ticker']
        daily = dict()
        for i, asset in enumerate(assets):
            out = self.get_value(asset.sid, sdate, edate)
            daily[asset.sid] = out.loc[:, supplements]
        return daily


class BcolzMinuteReader(BcolzReader):
    """
    Reader for data written by BcolzMinuteBarWriter

    Parameters
    ----------
    rootdir : string
        The root directory containing the metadata and asset bcolz
        directories.

    default_ohlc_ratio : int, optional
        The default ratio by which to multiply the pricing data to
        convert from floats to integers that fit within np.uint32. If
        ohlc_ratios_per_sid is None or does not contain a mapping for a
        given sid, this ratio is used. Default is OHLC_RATIO (100).

    minutes_per_day : int
        The number of minutes per each period. Defaults to 390, the mode
        of minutes in NYSE trading days.

    """
    def __init__(self):
        self._root_dir = os.path.join(TdxDir, 'minute')

    @property
    def data_frequency(self):
        return "minute"

    def get_spot_value(self, dt, asset, fields):
        """
        :param dt: str '%Y-%m-%d'
        :param asset: Asset
        :param fields: list
        :return:
        """
        start_dts = transfer_to_timestamp(dt + '09:30:00')
        end_dts = transfer_to_timestamp(dt + '15:00:00')
        minutes = self.get_value(asset.sid, start_dts, end_dts)
        return minutes.loc[:, fields + 'ticker']


class BcolzDailyReader(BcolzReader):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    Parameters
    ----------
    _rootdir : bcolz.ctable path
        The ctable contaning the pricing data, with attrs corresponding to the
        Attributes list below.
    calendar_name: str
        String identifier of trading _calendar used (ie, "China").

    Attributes:

    The data in these columns is interpreted as follows:
    - Price columns ('open', 'high', 'low', 'close') are interpreted as 100*
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    """
    def __init__(self):
        self._root_dir = os.path.join(TdxDir, 'daily')

    @property
    def data_frequency(self):
        return "daily"

    def get_spot_value(self, date, asset, fields):
        kline = self.get_value(asset.sid, date, date)
        return kline.loc[:, fields + 'trade_dt']
