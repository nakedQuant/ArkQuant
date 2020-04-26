# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""

from abc import ABC

class BarReader(ABC):

    def data_frequency(self):
        pass

    @abstractmethod
    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        pass

    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        pass

    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        pass

    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        pass

    @abstractmethod
    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        pass

    @abstractmethod
    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest minute on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded minute.
        dt : pd.Timestamp
            The minute at which to start searching for the last traded minute.

        Returns
        -------
        last_traded : pd.Timestamp
            The dt of the last trade for the given asset, using the input
            dt as a vantage point.
        """
        pass

    @abstractmethod
    def _dt_window_size(self, start_dt, end_dt):
        pass


from sqlalchemy import select,and_,cast ,Numeric,desc,Integer

class Core:
    def __init__(self):

        self.db = DataLayer()
        self.tables = self.db.metadata.tables

    def _proc(self,ins):
        rp = self.db.engine.execute(ins)
        return rp

    def load_kline(self,start_date,end_date,asset,tbl):
        """获取Stock 、 ETF 、Index 、Convertible Hk"""
        tbl = self.tables[tbl]
        if asset:
            ins = select([tbl.c.trade_dt,tbl.c.code,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                        cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,3)).label('low'),
                         cast(tbl.c.volume,Numeric(15,0)).label('volume')]). \
                where(and_(tbl.c.code ==asset,tbl.c.trade_dt.between(start_date, end_date)))
        else:
            ins = select([tbl.c.trade_dt,tbl.c.code,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                        cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,2)).label('low'),
                         cast(tbl.c.volume,Numeric(15,0)).label('volume')]).\
                        where(tbl.c.trade_dt.between(start_date,end_date))
        rp = self._proc(ins)
        kline = [[r.trade_dt,r.code,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        return kline

    def load_ashare_kline_offset(self,dt,window,asset):
        tbl = self.tables['asharePrice']
        ins = select([tbl.c.trade_dt,cast(tbl.c.open,Numeric(10,2)).label('open'),cast(tbl.c.close,Numeric(12,2)).label('close'),
                    cast(tbl.c.high,Numeric(10,2)).label('high'),cast(tbl.c.low,Numeric(10,3)).label('low'),
                     cast(tbl.c.volume,Numeric(15,0)).label('volume')]).\
                    where(and_(tbl.c.code == asset,tbl.c.trade_dt <= dt))
        ins = ins.order_by(desc(tbl.c.trade_dt))
        ins = ins.limit(window)
        rp = self._proc(ins)
        kline = [[r.trade_dt,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        return kline

    def load_stock_basics(self,asset):
        """ 股票基础信息"""
        table = self.tables['ashareInfo']
        if asset:
            ins = select([table]).where(table.c.代码 == asset)
        else:
            ins = select([table])
        rp = self._proc(ins)
        basics = rp.fetchall()
        return basics

    def load_convertible_basics(self,asset):
        """ 可转债基础信息"""
        table = self.tables['convertibleDesc']
        ins = select([table]).where(table.c.bond_id == asset)
        rp = self._proc(ins)
        basics = rp.fetchall()
        return basics[0]

    def load_equity_structure(self,asset):
        """
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = self.tables['ashareEquity']
        ins = select([table.c.代码,table.c.变动日期,table.c.公告日期,cast(table.c.总股本,Numeric(20,3)).label('总股本'),
                     cast(table.c.流通A股,Numeric(20,3)),cast(table.c.限售A股,Numeric(20,3)),cast(table.c.流通B股,Numeric(20,3)),cast(table.c.流通H股,Numeric(20,3))]).where(table.c.代码 == asset)
        rp = self._proc(ins)
        equtiy = rp.fetchall()
        return equtiy


    def load_ashare_mkv(self,sdate,edate,asset):
        """股票流通市值、总市值、B股市值，H股市值"""
        table = self.tables['ashareValue']
        if asset:
            ins = select([table.c.trade_dt,table.c.code,cast(table.c.mkt,Numeric(20,5)),cast(table.c.cap,Numeric(20,5)),cast(table.c.strict,Numeric(20,5)),cast(table.c.hk,Numeric(20,5))]).where\
                (and_(table.c.code == asset,table.c.trade_dt.between(sdate,edate)))
        else:
            ins = select([table.c.trade_dt,table.c.code,cast(table.c.mkt,Numeric(20,5)),cast(table.c.cap,Numeric(20,5)),cast(table.c.strict,Numeric(20,5)),cast(table.c.hk,Numeric(20,5))]).where\
                (table.c.trade_dt.between(sdate,edate))
        rp = self._proc(ins)
        market_value = rp.fetchall()
        return market_value

    def load_ashare_holdings(self,sdate,edate,asset):
        """股东持仓变动"""
        table = self.tables['ashareHolding']
        if asset:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (and_(table.c.代码 == asset,table.c.变动截止日.between(sdate,edate)))
        else:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (table.c.变动截止日.between(sdate,edate))
        rp = self._proc(ins)
        margin = rp.fetchall()
        return margin

    def load_market_margin(self,start,end):
        """A股市场的融资融券情况"""
        table = self.tables['marketMargin']
        ins = select([table.c.交易日期,cast(table.c.融资余额,Integer),cast(table.c.融券余额,Integer),cast(table.c.融资融券总额,Integer),cast(table.c.融资融券差额,Integer)]).where\
            (table.c.交易日期.between(start,end))
        rp = self._proc(ins)
        margin = rp.fetchall()
        return margin

    def load_stock_status(self,asset):
        """获取股票状态 : 退市、暂时上市"""
        table = self.tables['ashareStatus']
        if asset:
            ins = select([table.c.delist_date,table.c.status]).where(table.c.code == asset)
        else:
            ins = select([table.c.delist_date,table.c.status])
        rp = self._proc(ins)
        res = rp.fetchall()
        return res



import os,datetime,pandas as pd,json,bcolz
from functools import partial
from decimal import Decimal
from gateWay.spider.ts_engine import TushareClient
from gateWay.spider import spider_engine
from env.common import XML

class BarReader:

    def __init__(self):
        self.loader = Core()
        self.ts = TushareClient()
        self.extra = spider_engine.ExtraOrdinary()

    def _verify_fields(self,f,asset):
        """如果asset为空，fields必须asset"""
        field = f.copy()
        if not isinstance(field,list):
            raise TypeError('fields must be list')
        elif asset is None:
            field.append('code')
        return field

    def load_stock_kline(self,sdate, edate,fields,asset):
        """
            返回特定时间区间日股票K线
        """
        fields = self._verify_fields(fields,asset)
        kline = self.loader.load_kline(sdate, edate, asset, 'asharePrice')
        kline = pd.DataFrame(kline,columns = ['trade_dt','code','open','close','high','low','volume'])
        kline.index = kline.loc[:,'trade_dt']
        kl_pd = kline.loc[:,fields]
        return kl_pd

    def load_kl_offset(self,asset,dt,window):
        """返回股票特定时间偏移的K线"""
        offset_kline = self.loader.load_ashare_kline_offset(dt,window,asset)
        offset_kl = pd.DataFrame(offset_kline,columns = ['trade_dt','open','close','high','low','volume'])
        return offset_kl

    def load_hk_kline(self, sdate, edate,fields, asset):
        """返回港股Kline"""
        fields = self._verify_fields(fields,asset)
        hk = self.loader.load_kline(sdate, edate, asset, 'hkPrice')
        hk_kline = pd.DataFrame(hk,columns = ['trade_dt','code','open','close','high','low','volume'])
        hk_kline.index = hk_kline.loc[:,'trade_dt']
        hk_pd = hk_kline.loc[:,fields]
        return hk_pd

    def load_etf_kline(self, sdate, edate,fields,asset):
        """
            返回特定时间区间日ETF k线
        """
        fields = self._verify_fields(fields,asset)
        etf = self.loader.load_kline(sdate, edate, asset, 'fundPrice')
        etf_kline = pd.DataFrame(etf,columns = ['trade_dt','code','open','close','high','low','volume'])
        etf_kline.index = etf_kline.loc[:,'trade_dt']
        etf_pd = etf_kline.loc[:,fields]
        return etf_pd

    def load_convertible_kline(self, sdate, edate,fields,asset):
        """
            返回特定时间区间日可转债K线
        """
        fields = self._verify_fields(fields,asset)
        convertible = self.loader.load_kline(sdate, edate, asset,'convertiblePrice')
        convertible_kline = pd.DataFrame(convertible,columns = ['trade_dt','code','open','close','high','low','volume'])
        convertible_kline.index = convertible_kline.loc[:,'trade_dt']
        convertible_pd = convertible_kline.loc[:,fields]
        return convertible_pd

    def load_minute_kline(self, sid, window):
        dt= datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        if window:
            filename = ('-').join([dt,str(window), sid])
        else:
            filename = ('-').join([dt, sid])
        if filename not in os.listdir(XML.pathCsv.value):
            spider_engine.Astock.download_ticks(sid, window)
        # 读取csv 数据
        minute_kline = self.load_prices_from_csv(filename)
        return minute_kline

    def load_5d_minute_hk(self, h_code):
        """
            获取港股5日分钟线
            列名 -- ticker price volume
        """
        raw = self.extra.download_5d_minute_hk(h_code)
        return raw

class BarWriter:

    def __init__(self, path):

        self.sid_path = path

    def _write_csv(self, data):
        """
            dump to csv
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.sid_path)
        else:
            with open(self.sid_path, mode='w') as file:
                if isinstance(data, str):
                    file.write(data)
                else:
                    for chunk in data:
                        file.write(chunk)

    def _init_hdf5(self, frames, _complevel=5, _complib='zlib'):
        if isinstance(frames, json):
            frames = json.dumps(frames)
        with pd.HDFStore(self.sid_path, 'w', complevel=_complevel, complib=_complib) as store:
            panel = pd.Panel.from_dict(frames)
            panel.to_hdf(store)
            panel = pd.read_hdf(self.sid_path)
        return panel

    def _init_ctable(self, raw):
        """
            Obtain 、Create 、Append、Attr empty ctable for given path.
            addcol(newcol[, name, pos, move])	Add a new newcol object as column.
            append(cols)	Append cols to this ctable -- e.g. : ctable
            Flush data in internal buffers to disk:
                This call should typically be done after performing modifications
                (__settitem__(), append()) in persistence mode. If you don’t do this,
                you risk losing part of your modifications.

        """
        ctable = bcolz.ctable(rootdir=self.sid_path, columns=None, names= \
            ['open', 'high', 'low', 'close', 'volume'], mode='w')

        if isinstance(raw, pd.DataFrame):
            ctable.fromdataframe(raw)
        elif isinstance(raw, dict):
            for k, v in raw.items():
                ctable.attrs[k] = v
        elif isinstance(raw, list):
            ctable.append([raw])
        #
        ctable.flush()

    @staticmethod
    def load_prices_from_ctable(file):
        """
            bcolz.open return a carray/ctable object or IOError (if not objects are found)
            ‘r’ for read-only
            ‘w’ for emptying the previous underlying data
            ‘a’ for allowing read/write on top of existing data
        """
        sid_path = os.path.join(XML.CTABLE, file)
        table = bcolz.open(rootdir=sid_path, mode='r')
        df = table.todataframe(columns=[
            'open',
            'high',
            'low',
            'close',
            'volume'
        ])
        return df