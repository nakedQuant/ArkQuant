# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import namedtuple
from abc import ABC , abstractmethod
import struct,pandas as pd,bcolz,numpy as np,os,glob,datetime

OHLC_RATIO = 100


class BcolzWriter(ABC):

    def sidpath(self, sid):
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
        return os.join(self._rootdir, sid_subdir)

    def _init_metadata(self,path):
        #初始化
        metadata = namedtuple('metadata','start_session end_session ohlc_ratio roodir')
        #定义属性
        metadata.start_session = None
        metadata.end_session = self._end_session
        metadata.ohlc_ratio = self._default_ohlc_ratio
        metadata.rootdir = path
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
            names = self.COL_NAMES,
            mode='w',
        )
        table.flush()
        table.attrs['metadata'] = self._init_metadata(path)
        return table

    def set_sid_attrs(self, sid, **kwargs):
        """Write all the supplied kwargs as attributes of the sid's file.
        """
        table = self._ensure_ctable(sid)
        for k, v in kwargs.items():
            table.attrs[k] = v

    def _ensure_ctable(self, sid):
        """Ensure that a ctable exists for ``sid``, then return it."""
        sidpath = self.sidpath(sid)
        if not os.path.exists(sidpath):
            return self._init_ctable(sidpath)
        return bcolz.ctable(rootdir=sidpath, mode='a')

    @staticmethod
    def _normalize_date(source):
        source['year'] = source['dates'] // 2048 + 2004
        source['month'] = (source['dates'] % 2048) // 100
        source['day'] = (source['dates'] % 2048) % 100
        source['hour'] = source['sub_dates'] // 60
        source['minutes'] = source['sub_dates'] % 60
        source['ticker'] = source.apply(lambda x: pd.Timestamp(
            datetime.datetime(int(x['year']), int(x['month']), int(x['day']),
                              int(x['hour']), int(x['minutes']))),
                                axis=1)
        source['timestamp'] = source['ticker'].apply(lambda x: x.timestamp())
        return source.loc[:, ['timestamp', 'open', 'high', 'low', 'close', 'amount', 'volume']]

    @abstractmethod
    def retrieve_data_from_tdx(self, path):

        raise NotImplementedError()

    @abstractmethod
    def _write_internal(self, sid,data):
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

    def write_sid(self, sid,appendix):
        """
        Write a stream of minute data.
        :param sid: asset type
        :param appendix: .01 / .5 / .day
        :return: dataframe
        """
        path = os.path.join(self._source_dir,sid + appendix)
        try:
            data = self.retrieve_data_from_tdx(path)
        except IOError:
            print('tdx path is not correct')
        #
        self._write_internal(sid, data)

    def truncate(self,size = 0 ):
        """Truncate data when size = 0"""
        glob_path = os.path.join(self._rootdir,"*.bcolz")
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
    COL_NAMES = frozenset(['ticker','open', 'high', 'low', 'close','amount','volume'])

    def __init__(self,
                 rootdir,
                 tdx_minutes_dir,
                 default_ratio = OHLC_RATIO):

        self._rootdir = rootdir
        self._source_dir = tdx_minutes_dir
        self._default_ohlc_ratio = default_ratio
        self._end_session = pd.Timestamp('1990-01-01').timestamp()

    def __setattr__(self, key, value):

        raise NotImplementedError()

    def retrieve_data_from_tdx(self,path):
        """解析通达信数据"""
        with open(path, 'rb') as f:
            buf = f.read()
            size = int(len(buf) / 32)
            data = []
            for num in range(size):
                idx = 32 * num
                struct_line = struct.unpack('HhIIIIfii', buf[idx:idx + 32])
                data.append(struct_line)
            dataframe = pd.DataFrame(data, columns=['dates', 'sub_dates', 'open',
                                                    'high', 'low', 'close', 'amount',
                                                    'volume', 'appendix'])
            ticker = self._normalize_date(dataframe)
            return ticker

    def _write_internal(self, sid,data):
        table = self._ensure_ctable(sid)
        #剔除重复的
        metadata = table.attr['metadata']
        dataframes = data[data['timestamp'] > metadata.end_session]
        if dataframes:
            table.append(dataframes)
            #更新metadata
            metadata.end_session = dataframes['timestamp'].max()
            if not metadata.start_session :
                metadata.start_session = dataframes['timestamp'].min()
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
    COL_NAMES = frozenset(['trade_dt','open', 'high', 'low', 'close','amount','volume'])

    def __init__(self,
                 rootdir,
                 txn_daily_dir):

        self._rootdir = rootdir
        self._source_dir = txn_daily_dir
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
            raw = pd.DataFrame(data,columns=['trade_dt', 'open', 'high', 'low',
                                            'close', 'amount', 'volume', 'appendix'])
            return raw

    def _write_internal(self, sid,data):
        table = self._ensure_ctable(sid)
        #剔除重复的
        metadata = table.attr['metadata']
        dataframes = data[data['trade_dt'] > metadata.end_session]
        if dataframes:
            table.append(dataframes)
            #更新metadata
            metadata.end_session = dataframes['trade_dt'].max()
            if not metadata.start_session :
                metadata.start_session = dataframes['trade_dt'].min()
            table.attrs['metadata'] = metadata
            # data in memory to disk
            table.flush()


#reader

default = frozenset(['open', 'high', 'low', 'close','amount','volume'])


class BcolzReader(ABC):

    @property
    def data_frequency(self):
        raise NotImplementedError()

    @property
    def calenar(self):
        return self._calendar

    def _window_dt(self,dt,length):
        _forward_dt = self.calendar._roll_forward(dt,length)
        return _forward_dt

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

    @abstractmethod
    def get_resampled(self,*args):
        """
         List of DatetimeIndex representing the minutes to exclude because
         of early closes.
        """
        raise NotImplementedError()

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

    @abstractmethod
    def get_spot_value(self,dt,sids,fields):
        """
            retrieve data on dt
        """
        raise NotImplementedError()

    @abstractmethod
    def load_raw_arrays(self,*args):
        """
        bcolz.open return a carray/ctable object or IOError (if not objects are found)
            ‘r’ for read-only
            ‘w’ for emptying the previous underlying data
            ‘a’ for allowing read/write on top of existing data

        Parameters
        ----------
        start_dt: Timestamp
           Beginning of the window range.
        end_dt: Timestamp
           End of the window range.
        sids : list of int
           The asset identifiers in the window.
        fields : list of str
           'open', 'high', 'low', 'close', or 'volume'

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        raise NotImplementedError()


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
        self._seconds_per_day = 24 * 60 * 60

    @property
    def data_frequency(self):
        return "minute"

    def get_resampled(self,sessions,dts,sids,field = default):
        """
            select specific dts minutes ,e,g --- 9:30,10:30
        """
        resample_tickers = {}
        arrays = self.load_raw_arrays(sessions[0],sessions[1],sids,field)
        for sid,raw in arrays.items():
            ticker_seconds = dts.split(':')[0] * 60 * 60 + dts.split(':')[0] * 60
            data = raw.fetchwhere("(timestamp - {0}) % {1} == 0".format(ticker_seconds,self._seconds_per_day))
            resample_tickers[sid] = data
        return resample_tickers

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

    def get_resampled(self,sessions,frequency,sids,field = default):
        """
            select specific dts minutes ,e,g --- 9:30,10:30
        """
        resampled = {}
        arrays = self.load_raw_arrays(sessions[0],sessions[1],sids,field)
        pds = [dt.strftime('%Y%m%d') for dt in pd.date_range(sessions[0],sessions[1],freq = frequency)]
        for sid,raw in arrays.items():
            resampled[sid] = raw.loc[pds,:]
        return resampled

    def get_spot_value(self,dt,sids,fields = default):
        minutes = self.load_raw_arrays(dt, dt, sids, fields)
        return minutes

    def load_raw_arrays(self,end,window,assets,columns = default):
        sdate = self._window_dt(end,window)
        supplement_fields = columns + ['trade_dt']
        daily = []
        for i, sid in enumerate(assets):
            out = self.get_value(sid,sdate,end)
            daily.append(out.loc[:,supplement_fields])
        return daily


__all__ = [BcolzDailyBarWriter,BcolzDailyReader,BcolzMinuteBarWriter,BcolzMinuteReader]