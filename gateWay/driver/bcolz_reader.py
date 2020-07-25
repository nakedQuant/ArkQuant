# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd,bcolz,os,datetime
from .bar_reader import BarReader

default = frozenset(['open', 'high', 'low', 'close','amount','volume'])


class BcolzReader(BarReader):

    @staticmethod
    def _to_timestamp(dt,last = False):
        if not isinstance(dt,pd.Timestamp):
            try:
                stamp = pd.Timestamp(dt)
            except Exception as e:
                raise TypeError('cannot tranform %r to timestamp due to %s'%(dt,e))
        else:
            stamp = dt
        final = datetime.datetime(stamp.year,stamp.month,stamp.day,hour = 15,minute = 0,second=0) if last else \
            datetime.datetime(stamp.year,stamp.month,stamp.day,hour = 9,minute = 30,second=0)
        return final

    def get_sid_attr(self, sid):
        sid_path = '{}.bcolz'.format(sid)
        root = os.path.join(self._rootdir, sid_path)
        return root

    def _read_bcolz_data(self, sid):
        """cparams(clevel=5, shuffle=1, cname='lz4', quantize=0)"""
        rootdir = self.get_sid_attr(sid)
        table = bcolz.open(rootdir=rootdir, mode='r')
        return table

    def get_value(self, sid,sdate,edate):
        """
        Retrieve the pricing info for the given sid, dt, and field.

        Parameters
        ----------
        sid : int
            Asset identifier.
        dt : datetime-like
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
        assert meta.end_session >= sdate,('%r exceed metadata end_session'%sdate)
        #获取数据
        if self.data_frequency == 'minute':
            start = self.transfer_to_timestamp(sdate)
            end = self.transfer_to_timestamp(edate, last=True)
            condition = '({0} <= timestamp) & (timestamp <= {1)'.format(start.timestamp(), end.timestamp())
        else:
            condition = '({0} <= trade_dt) & (trade_dt <= {1)'.format(sdate, edate)
        raw = table.fetchwhere(condition)
        dataframe = pd.DataFrame(raw)
        #调整系数
        inverse_ratio = 1 / meta.ohlc_ratio
        scale = dataframe.apply(lambda x : [ x[col] * inverse_ratio
                                for col in ['open','high','low','close']],
                                axis = 1)
        scale.set_index('ticker',inplace= True) if 'ticker' in scale.columns else scale.set_index('trade_dt',inplace = True)
        return scale


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
    def __init__(self,
                 rootdir,
                 trading_calendar):

        self._rootdir = rootdir
        self._calendar = trading_calendar

    @property
    def data_frequency(self):
        return "minute"

    def get_spot_value(self,dt,sids,fields = default):
        dts = self.transfer_to_timestamp(dt)
        minutes = self.load_raw_arrays(dts,0,sids,fields)
        return minutes

    def load_raw_arrays(self,end_dt,window,sids,fields = default):
        sdate = self._window_dt(end_dt,window)
        supplement_fields = fields + ['ticker']
        minutes = []
        for i, sid in enumerate(sids):
            out = self.get_value(sid,sdate,end_dt)
            minutes.append(out.loc[:,supplement_fields])
        return minutes


class BcolzDailyReader(BcolzReader):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    Parameters
    ----------
    _rootdir : bcolz.ctable path
        The ctable contaning the pricing data, with attrs corresponding to the
        Attributes list below.
    calendar_name: str
        String identifier of trading calendar used (ie, "China").

    Attributes:

    The data in these columns is interpreted as follows:
    - Price columns ('open', 'high', 'low', 'close') are interpreted as 100*
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    """
    def __init__(self,
                 rootdir,
                 trading_calendar):

        self._rootdir = rootdir
        self._calendar = trading_calendar

    @property
    def data_frequency(self):
        return "daily"

    def get_spot_value(self,dt,asset):
        minutes = self.load_raw_arrays(dt, dt, asset)
        return minutes

    def load_raw_arrays(self,end,window,assets,columns = default):
        sdate = self._window_dt(end,window)
        supplement_fields = columns + ['trade_dt']
        daily = []
        for i, sid in enumerate(assets):
            out = self.get_value(sid,sdate,end)
            daily.append(out.loc[:,supplement_fields])
        return daily


__all__ = [BcolzDailyReader,BcolzMinuteReader]