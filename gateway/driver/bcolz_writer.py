# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import struct, pandas as pd, bcolz, numpy as np, os, glob, datetime, json
from abc import ABC, abstractmethod
from utils.dt_utilty import normalize_date

Num = 2
OHLC_RATIO = 10000
# 解析通达信数据存储位置
BcolzDir = r'E:\bcolz'
BcolzMinuteFields = ['ticker', 'open', 'high', 'low', 'close', 'amount', 'volume']
BcolzDailyFields = ['trade_dt', 'open', 'high', 'low', 'close', 'amount', 'volume']
# 配置bcolz
bcolz.set_nthreads(Num * bcolz.detect_number_of_cores())
# Print all the versions of packages that bcolz relies on.
bcolz.print_versions()


class BcolzWriter(ABC):
    """
        The defaults for parameters used in compression (dict).
        The default is {‘clevel’: 5, ‘shuffle’: True, ‘cname’: ‘lz4’, quantize: 0}.
        classbcolz.cparams(clevel=None, shuffle=None, cname=None, quantize=None)

        Parameters:
        clevel : int (0 <= clevel < 10)
        The compression level.
        shuffle : int
        The shuffle filter to be activated. Allowed values are bcolz.NOSHUFFLE (0), bcolz.SHUFFLE (1) and bcolz.BITSHUFFLE (2). The default is bcolz.SHUFFLE.
        cname : string (‘blosclz’, ‘lz4’, ‘lz4hc’, ‘snappy’, ‘zlib’, ‘zstd’)
        Select the compressor to use inside Blosc.
        quantize : int (number of significant digits)
        Quantize data to improve (lossy) compression. Data is quantized using np.around(scale*data)/scale, where scale is 2**bits,
        and bits is determined from the quantize value. For example, if quantize=1, bits will be 4. 0 means that the quantization is disabled.
        In case some of the parameters are not passed, they will be
        set to a default (see `setdefaults()` method).
        target --- tranform tdx files to bcolz
    """

    def _init_attr(self, metadata, path):
        # 初始化
        # metadata = defaultdict(None)
        # 定义属性
        metadata.attrs['start_session'] = None
        metadata.attrs['end_session'] = None
        metadata.attrs['ohlc_ratio'] = self._default_ohlc_ratio
        metadata.attrs['root_dir'] = path
        # serialized = json.dumps(metadata)
        # print('serialized', serialized)
        # return serialized
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
        bcolz_dir = os.path.dirname(path)
        print('bcolz_dir', bcolz_dir)
        if not os.path.exists(bcolz_dir):
            os.makedirs(bcolz_dir)
            print('path', path)
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
            names=self._bcolz_fields,
            mode='w',
        )
        table.flush()
        table = self._init_attr(table, path)
        # table.attrs['metadata'] = self._init_metadata(path)
        return table

    def _sid_path(self, sid):
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
        return os.path.join(self._root_dir, sid_subdir)

    def _ensure_ctable(self, sid):
        """Ensure that a ctable exists for ``sid``, then return it."""
        sid_path = self._sid_path(sid)
        print('sid_path', sid_path)
        if not os.path.exists(sid_path):
            return self._init_ctable(sid_path)
        return bcolz.ctable(rootdir=sid_path, mode='a')

    # def set_sid_attrs(self, sid, **kwargs):
    #     """Write all the supplied kwargs as attributes of the sid's file.
    #     """
    #     table = self._ensure_ctable(sid)
    #     for k, v in kwargs.items():
    #         table.attrs[k] = v

    def glob_tdx_files(self):
        # dir_ = r'D:\通达信-1m\*'
        sh_path = os.path.join(self._tdx_dir, r'*\vipdoc\*\minline\sh6*')
        sz_path = os.path.join(self._tdx_dir, r'*\vipdoc\*\minline\sz[0|3]*')
        sh_files = glob.glob(sh_path)
        sz_files = glob.glob(sz_path)
        tdx_file_paths = sh_files + sz_files
        print('files', len(tdx_file_paths), tdx_file_paths[:10])
        return tdx_file_paths

    @abstractmethod
    def retrieve_data_from_tdx(self, path):
        raise NotImplementedError()

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

    def _write_sid(self, path):
        """
        Write a stream of minute data.
        :param sid: asset type (sh / sz)
        :param appendix: .01 / .5 / .day
        :return: dataframe
        """
        # path = os.path.join(self._tdx_dir, ('.').join([sid, appendix]))
        try:
            data = self.retrieve_data_from_tdx(path)
        except IOError:
            print('tdx path:%s is not correct' %path)
        else:
            sid = os.path.basename(path)
            self._write_internal(sid[:-3], data)

    def write(self):
        paths = self.glob_tdx_files()
        for p in paths:
            self._write_sid(p)

    def truncate(self, size=0):
        """Truncate data when size = 0"""
        glob_path = os.path.join(self._root_dir, "*.bcolz")
        print('glob_path', glob_path)
        sid_paths = sorted(glob.glob(glob_path))
        for sid_path in sid_paths:
            try:
                table = bcolz.open(rootdir=sid_path)
            except IOError:
                continue
            print('before resize', table.len)
            # Resize the instance to have nitems
            table.resize(size)
            print('after resize', table.len)


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
    def __init__(self,
                 minutes_dir,
                 default_ratio=OHLC_RATIO):
        # tdx_dir --- 通达信数据所在
        self._tdx_dir = minutes_dir
        self._default_ohlc_ratio = default_ratio
        self._root_dir = os.path.join(BcolzDir, 'minute')
        self._bcolz_fields = BcolzMinuteFields
        self.c_table = bcolz.ctable(columns=BcolzMinuteFields)

    def retrieve_data_from_tdx(self, path):
        """解析通达信数据"""
        with open(path, 'rb') as f:
            buf = f.read()
            size = int(len(buf) / 32)
            data = []
            for num in range(size):
                idx = 32 * num
                line = struct.unpack('HhIIIIfii', buf[idx:idx + 32])
                data.append(line)
            frame = pd.DataFrame(data, columns=['dates', 'sub_dates', 'open',
                                                'high', 'low', 'close', 'amount',
                                                'volume', 'appendix'])
            frame = normalize_date(frame)
            ticker_frame = frame.loc[:, BcolzMinuteFields]
            return ticker_frame

    def _write_internal(self, sid, data):
        table = self._ensure_ctable(sid)
        # 剔除重复的
        start_session = table.attrs['start_session']
        end_session = table.attrs['end_session']
        data['ticker'] = data['ticker'].apply(lambda x: x.timestamp())
        frame = data[data['ticker'] > end_session] if end_session else data
        if len(frame):
            ctable_frame = self.c_table.fromdataframe(frame)
            table.append(ctable_frame)
            # 更新metadata
            table.attrs['end_session'] = frame['ticker'].max()
            if not start_session:
                table.attrs['start_session'] = frame['ticker'].min()
            # data in memory to disk
            table.flush()
        print(table.len)
        print('metadata', table.attrs)
        # test
        # df = table.todataframe()
        # print('df', df)


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
    def __init__(self,
                 daily_dir,
                 default_ratio=OHLC_RATIO):
        self._tdx_dir = daily_dir
        self._root_dir = os.path.join(BcolzDir, 'daily')
        self._default_ohlc_ratio = default_ratio
        self._bcolz_fields = BcolzDailyFields
        self.c_table = bcolz.ctable(columns=BcolzDailyFields)

    def retrieve_data_from_tdx(self, path):
        with open(path, 'rb') as f:
            buf = f.read()
            size = int(len(buf) / 32)
            data = []
            for num in range(size):
                idx = 32 * num
                line = struct.unpack('IIIIIfII', buf[idx:idx + 32])
                data.append(line)
            frame = pd.DataFrame(data, columns=['trade_dt', 'open', 'high', 'low',
                                                'close', 'amount', 'volume', 'appendix'])
            # transform to Timestamp
            frame['trade_dt'] = frame['trade_dt'].apply(lambda x: str(x))
            return frame

    def _write_internal(self, sid, data):
        table = self._ensure_ctable(sid)
        # 剔除重复的
        start_session = table.attrs['start_session']
        end_session = table.attrs['end_session']
        frame = data[data['trade_dt'] > end_session] if end_session else data
        if not frame.empty:
            print('daily frame', frame.head())
            ctable_frame = self.c_table.fromdataframe(frame)
            table.append(ctable_frame)
            # 更新metadata
            table.attrs['end_session'] = frame['trade_dt'].max()
            if not start_session:
                table.attrs['start_session'] = frame['trade_dt'].min()
            # data in memory to disk
            table.flush()
        print(table.len)
        print('metadata', table.attrs)
        # df = table.todataframe()
        # print('df', df)


if __name__ == '__main__':

    # glob different code name
    # files = glob.glob('^(sz|sh)[0-9]{6}.01$', recursive=True)
    # files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\*')
    # files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\sz[0|3]*')
    # files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\sz0*')
    # files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\sz3*')
    # sh_files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\sh6*')
    # sz_files = glob.glob(r'D:\通达信-1m\*\vipdoc\*\minline\sz[0|3]*')
    # files = sz_files + sh_files
    # print('files', len(files), files[:10])
    # # initialize all time stamp
    # start_date = pd.Timestamp('20040401').to_pydatetime()
    # end_date = datetime.datetime.now()
    # session = pd.date_range(start_date, end_date, freq='1m')
    # print('session', session)
    # tdx_names = [datetime.datetime.strftime(s, '%Y%m') for s in session]
    # print('formatted datetime', tdx_names)
    # t = datetime.datetime.now()
    # for tdx_name in tdx_names:
    #     source_dir = r'D:\通达信-1m\{}\vipdoc\sh\minline'.format(tdx_name)
    #     print('source_dir', source_dir)
    #     w1 = BcolzMinuteBarWriter(source_dir)
    #     w1.write_sid('sh000001', '01')
    #     # w1.truncate(size=0)
    # print('elapsed time', datetime.datetime.now() - t)
    # elapsed time 0:01:16.533000 --- set_threads
    # elapsed time 0:01:17.158000 --- no set
    # source_dir = r'E:\通达信-1d\202008\vipdoc\sh\lday'
    # w2 = BcolzDailyBarWriter(source_dir)
    # w2.write_sid('sh000001', 'day')

    dir_ = r'D:\通达信-1m'
    w1 = BcolzMinuteBarWriter(dir_)
    w1.write()
