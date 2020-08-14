# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import namedtuple
from abc import ABC, abstractmethod
import struct, pandas as pd, bcolz, numpy as np, os, glob, datetime
from gateway.driver import TdxDir, OHLC_RATIO


__all__ = [
    'BcolzDailyBarWriter',
    'BcolzMinuteBarWriter'
]


class BcolzWriter(ABC):

    def _init_metadata(self, path):
        # 初始化
        metadata = namedtuple('metadata', 'start_session end_session ohlc_ratio root_dir')
        # 定义属性
        metadata.start_session = None
        metadata.end_session = self._end_session
        metadata.ohlc_ratio = self._default_ohlc_ratio
        metadata.root_dir = path
        return metadata

    def _init_ctable(self, path):
        """
        Create empty ctable for given path.
        Obtain 、Create 、Append、Attr empty ctable for given path.
        addcol(newcol[, name, pos, move])	Add a new newcol object as column.
        append(cols)	Append cols to this ctable -- e.g. : ctable
        Flush data in internal buffers to disk:
        This call should typically be done after performing modifications
        (__settitem__(), append()) in persistence mode. If you don’t do this,
        you risk losing part of your modifications.

        Parameters
        ----------
        path : string
            The path to rootdir of the new ctable.
        """
        sid_dirname = os.path.dirname(path)
        if not os.path.exists(sid_dirname):
            os.makedirs(sid_dirname)
        initial_array = np.empty(0, np.uint32)
        table = bcolz.ctable(
            rootdir=path,
            columns=[
                initial_array,
                initial_array,
                initial_array,
                initial_array,
                initial_array,
                initial_array,
                initial_array,
            ],
            names=self.COL_NAMES,
            mode='w',
        )
        table.flush()
        table.attrs['metadata'] = self._init_metadata(path)
        return table

    def sid_path(self, sid):
        """
        Parameters
        ----------
        sid : int
            Asset identifier.

        Returns
        -------
        out : string
            Full path to the bcolz rootdir for the given sid.
        """
        sid_subdir = '{}.bcolz'.format(sid)
        return os.join(self._root_dir, sid_subdir)

    def _ensure_ctable(self, sid):
        """Ensure that a ctable exists for ``sid``, then return it."""
        sid_path = self.sid_path(sid)
        if not os.path.exists(sid_path):
            return self._init_ctable(sid_path)
        return bcolz.ctable(rootdir=sid_path, mode='a')

    def set_sid_attrs(self, sid, **kwargs):
        """Write all the supplied kwargs as attributes of the sid's file.
        """
        table = self._ensure_ctable(sid)
        for k, v in kwargs.items():
            table.attrs[k] = v

    @staticmethod
    def _normalize_date(raw):
        raw['year'] = raw['dates'] // 2048 + 2004
        raw['month'] = (raw['dates'] % 2048) // 100
        raw['day'] = (raw['dates'] % 2048) % 100
        raw['hour'] = raw['sub_dates'] // 60
        raw['minutes'] = raw['sub_dates'] % 60
        raw['ticker'] = raw.apply(lambda x: pd.Timestamp(
            datetime.datetime(int(x['year']), int(x['month']), int(x['day']),
                              int(x['hour']), int(x['minutes']))),
                                axis=1)
        raw['timestamp'] = raw['ticker'].apply(lambda x: x.timestamp())
        return raw.loc[:, ['timestamp', 'open', 'high', 'low', 'close', 'amount', 'volume']]

    @abstractmethod
    def retrieve_data_from_tdx(self, path):

        raise NotImplementedError()

    def write_sid(self, sid, appendix):
        """
        Write a stream of minute data.
        :param sid: asset type
        :param appendix: .01 / .5 / .day
        :return: dataframe
        """
        path = os.path.join(self._tdx_dir, sid + appendix)
        try:
            data = self.retrieve_data_from_tdx(path)
        except IOError:
            print('tdx path is not correct')
        #
        self._write_internal(sid, data)

    @abstractmethod
    def _write_internal(self, sid, data):
        """
        Internal method for `write_cols` and `write`.

        Parameters
        ----------
        sid : int
            The asset identifier for the data being written.
        data : dict of str -> np.array
            dict of market data with the following characteristics.
            keys are ('open', 'high', 'low', 'close', 'volume')
            open : float64
            high : float64
            low  : float64
            close : float64
            volume : float64|int64
        """
        raise NotImplementedError()

    def truncate(self, size=0):
        """Truncate data when size = 0"""
        glob_path = os.path.join(self._root_dir, "*.bcolz")
        sid_paths = sorted(glob(glob_path))
        for sid_path in sid_paths:
            try:
                table = bcolz.open(rootdir=sid_path)
            except IOError:
                continue
            table.resize(size)


class BcolzMinuteBarWriter(BcolzWriter):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.

    Parameters
    ----------
    rootdir : string
        Path to the root directory into which to write the metadata and
        bcolz subdirectories.
    tdx_min_dir : tdx minutes data path

    Notes
    -----
    Writes a bcolz directory for each individual sid, all contained within
    a root directory which also contains metadata about the entire dataset.

    Each individual asset's data is stored as a bcolz table with a column for
    each pricing field: (open, high, low, close, volume)

    The open, high, low, and close columns are integers which are 1000 times
    the quoted price, so that the data can represented and stored as an
    np.uint32, supporting market prices quoted up to the thousands place.

    volume is a np.uint32 with no mutation of the tens place.
    """
    COL_NAMES = frozenset(['ticker', 'open', 'high', 'low', 'close', 'amount', 'volume'])

    def __init__(self,
                 minutes_dir,
                 default_ratio=OHLC_RATIO):
        # tdx_dir --- 通达信数据所在
        self._tdx_dir = minutes_dir
        # 解析H5数据所在位置
        self._root_dir = os.path.join(TdxDir, 'minute')
        self._default_ohlc_ratio = default_ratio
        self._end_session = pd.Timestamp('1990-01-01').timestamp()

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def retrieve_data_from_tdx(self, path):
        """解析通达信数据"""
        with open(path, 'rb') as f:
            buf = f.read()
            size = int(len(buf) / 32)
            data = []
            for num in range(size):
                idx = 32 * num
                struct_line = struct.unpack('HhIIIIfii', buf[idx:idx + 32])
                data.append(struct_line)
            frame = pd.DataFrame(data, columns=['dates', 'sub_dates', 'open',
                                                    'high', 'low', 'close', 'amount',
                                                    'volume', 'appendix'])
            ticker = self._normalize_date(frame)
            return ticker

    def _write_internal(self, sid, data):
        table = self._ensure_ctable(sid)
        # 剔除重复的
        metadata = table.attr['metadata']
        frames = data[data['timestamp'] > metadata.end_session]
        if frames:
            table.append(frames)
            # 更新metadata
            metadata.end_session = frames['timestamp'].max()
            if not metadata.start_session:
                metadata.start_session = frames['timestamp'].min()
            table.attrs['metadata'] = metadata
            # data in memory to disk
            table.flush()


class BcolzDailyBarWriter(BcolzWriter):
    """
    Class capable of writing daily OHLCV data to disk in a format that can
    be read efficiently by BcolzDailyOHLCVReader.

    Parameters
    ----------
    rootdir : str
        The location where daily bcolz  exists
    txn_daily_dir : tdx daily data path

    See Also
    --------
    zipline.data.bcolz_daily_bars.BcolzDailyBarReader
    """
    COL_NAMES = frozenset(['trade_dt', 'open', 'high', 'low', 'close', 'amount', 'volume'])

    def __init__(self,
                 daily_dir):
        self._txn_dir = daily_dir
        self._root_dir = os.path.join(TdxDir, 'daily')
        self._end_session = '1990-01-01'

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def retrieve_data_from_tdx(self, path):
        with open(path, 'rb') as f:
            buf = f.read()
            size = int(len(buf) / 32)
            data = []
            for num in range(size):
                idx = 32 * num
                struct_line = struct.unpack('IIIIIfII', buf[idx:idx + 32])
                data.append(struct_line)
            raw = pd.DataFrame(data, columns=['trade_dt', 'open', 'high', 'low',
                                              'close', 'amount', 'volume', 'appendix'])
            return raw

    def _write_internal(self, sid, data):
        table = self._ensure_ctable(sid)
        #剔除重复的
        metadata = table.attr['metadata']
        frames = data[data['trade_dt'] > metadata.end_session]
        if frames:
            table.append(frames)
            #更新metadata
            metadata.end_session = frames['trade_dt'].max()
            if not metadata.start_session:
                metadata.start_session = frames['trade_dt'].min()
            table.attrs['metadata'] = metadata
            # data in memory to disk
            table.flush()
