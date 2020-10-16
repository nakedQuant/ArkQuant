# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, bcolz, os
from gateway.driver.bar_reader import BarReader
from gateway.driver.tools import transfer_to_timestamp
from gateway.driver import BcolzDir, OHLC_RATIO


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
        prefix_sid = 'sh' + sid if sid.startswith('6') else 'sz' + sid
        bcolz_file = '{}.bcolz'.format(prefix_sid)
        root = os.path.join(self._root_dir, bcolz_file)
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
        print('cparams', table.cparams)
        meta = table.attrs
        # apply functools
        # # test = init.eval('(ticker - 34200) / 86400')
        # 获取数据
        if self.data_frequency == 'minute':
            start = transfer_to_timestamp(sdate)
            print('start', sdate, start)
            assert meta['end_session'] >= start, ('%r exceed metadata end_session' % start)
            end = transfer_to_timestamp(edate) + 15 * 60 * 60
            print('end', edate, end)
            condition = '({0} <= ticker) & (ticker <= {1})'.format(start, end)
        else:
            assert meta['end_session'] >= sdate, ('%r exceed metadata end_session' % sdate)
            condition = "({0} <= trade_dt) & (trade_dt <= {1})".format(sdate, edate)
        frame = pd.DataFrame(table.fetchwhere(condition))
        if not frame.empty:
            frame.set_index('ticker', inplace=True) if 'ticker' in frame.columns \
                else frame.set_index('trade_dt', inplace=True)
            # 调整系数  原来的系数有问题（10000） --- 100
            # inverse_ratio = 1 / meta['ohlc_ratio']
            inverse_ratio = 1 / OHLC_RATIO
            # print('original frame', frame)
            frame.loc[:, ['open', 'high', 'low', 'close']] = frame.loc[:, ['open', 'high', 'low', 'close']] * inverse_ratio
            # transform timestamp to structtime based on the utc (UTC只是比北京时间提前了8个小时)
            # frame.index = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M') for i in frame.index] \
            #     if frame.index.name == 'ticker' else frame.index
        return frame

    def get_spot_value(self, dt, asset, fields):
        raise NotImplementedError()

    def get_stack_value(self, tbl_name, session):
        raise NotImplementedError(
            'minute reader cannot implement stack value '
            'because different sid has solo bcolz'
        )

    def load_raw_arrays(self, sessions, assets, columns):
        assert set(columns).issubset(self.default), 'unknown field'
        sdate, edate = sessions
        frame_dict = dict()
        for i, asset in enumerate(assets):
            out = self.get_value(asset.sid, sdate, edate)
            frame_dict[asset.sid] = out.loc[:, columns]
        return frame_dict


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
        self._root_dir = os.path.join(BcolzDir, 'minute')

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
        start_dts = transfer_to_timestamp(dt + ' 09:30:00')
        end_dts = transfer_to_timestamp(dt + ' 15:00:00')
        minutes = self.get_value(asset.sid, start_dts, end_dts)
        return minutes.loc[:, fields]


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
        self._root_dir = os.path.join(BcolzDir, 'daily')

    @property
    def data_frequency(self):
        return "daily"

    def get_spot_value(self, dt, asset, fields):
        kline = self.get_value(asset.sid, dt, dt)
        return kline.loc[:, fields]


# if __name__ == '__main__':
#
#     from gateway.asset.assets import Equity
#
#     session = ['2005-01-01', '2015-09-03']
#     equity = Equity('600000')
#     fields = ['open', 'close', 'volume']
#     minute_reader = BcolzMinuteReader()
#     spot_value = minute_reader.get_spot_value('2005-09-07', equity, fields)
#     import datetime
#     spot_value.index = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M') for i in spot_value.index]
#     print('spot_value', spot_value)
#     t = spot_value['close'] * spot_value['volume'] / spot_value['volume'].sum()
#     print('t', t.sum())
#     raw = minute_reader.load_raw_arrays(session, [equity], fields)
#     raw = minute_reader.load_raw_arrays(['2005-09-05', '2005-09-07'], [equity], fields)
#     print('raw', raw)
