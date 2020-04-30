# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import create_engine,MetaData,select,and_,cast ,Numeric ,Integer
import pandas as pd ,json ,os ,requests
from bs4 import BeautifulSoup
from abc import ABC , abstractmethod

from gateWay.spider.ts_engine import TsClient

ts = TsClient()


def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=1)
    if encoding:
        req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw



class Reader(ABC):

    @abstractmethod
    def _init_db(self,engine_path):
        engine = create_engine(engine_path)
        self.conn = engine.connect.execution_options(
                    isolation_level="READ UNCOMMITTED")
        metadata = MetaData(bind=engine)
        for table_name in engine.table_names():
            setattr(self,table_name,metadata.tables[table_name])


    @property
    def first_trading_day(self,sid):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        orm = select([self.equity_basics.c.initial_date]).where(self.equity_basics.c.sid == sid)
        first_dt = self.conn.execute(orm).scalar()
        return first_dt

    @property
    def get_last_traded_dt(self, asset):
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
        orm = select([self.symbol_delist.c.delist_date]).where(self.symbol_delist.c.sid == asset)
        rp = self.conn.execute(orm)
        dead_date = rp.scalar()
        return dead_date


class BarReader(Reader):

    def __init__(self,engine_path):
        super(BarReader,self)._init_db(engine_path)

    def _load_raw_arrays(self,start_date,end_date,asset,tbl):
        tbl = self.__getattribute__(tbl)
        orm_sql = select([tbl.c.trade_dt,tbl.c.code,
                      cast(tbl.c.open,Numeric(10,2)).label('open'),
                      cast(tbl.c.close,Numeric(12,2)).label('close'),
                      cast(tbl.c.high,Numeric(10,2)).label('high'),
                      cast(tbl.c.low,Numeric(10,3)).label('low'),
                      cast(tbl.c.volume,Numeric(15,0)).label('volume'),
                      cast(tbl.c.volume,Numeric(15,2)).label('amount')]). \
             where(and_(tbl.c.code == asset,
                        tbl.c.trade_dt.between(start_date, end_date)))
        rp = self.conn.execute(orm_sql)
        arrays = [[r.trade_dt,r.code,r.open,r.close,r.high,r.low,r.volume] for r in rp.fetchall()]
        kline = pd.DataFrame(arrays,columns = ['trade_dt','code','open','close','high','low','volume','amount'])
        return kline

    def load_daily_symbol(self,sdate, edate,fields,asset):
        """
            返回特定时间区间日股票K线
        """
        kline = self._load_raw_arrays(sdate,edate,asset,'symbol_price')
        kl_pd = kline.loc[:,fields]
        return kl_pd

    def load_daily_fund(self, sdate, edate,fields,asset):
        kline = self._load_raw_arrays(sdate,edate,asset,'fund_price')
        fund_kline = kline.loc[:,fields]
        etf_pd = fund_kline.loc[:,fields]
        return etf_pd

    def load_daily_bond(self, sdate, edate,fields,asset):
        """
            返回特定时间区间日可转债K线
        """
        kline = self._load_raw_arrays(sdate,edate,asset,'bond_price')
        bond_kline = kline.loc[:,fields]
        bond_pd = bond_kline.loc[:,fields]
        return bond_pd

    def load_daily_dual(self, sdate, edate,fields, asset):
        """
            A H 同时上市
        """
        kline = self._load_raw_arrays(sdate, edate, asset, 'dual_symbol_price')
        hk_kline = kline.loc[:, fields]
        return hk_kline

    def load_minutes(self, sid,lag = 0):
        req_sid = '0.' + sid if sid.startswith('6') else '1.' + sid
        if lag is None:
            # 获取当日日内数据
            html_m = 'http://push2.eastmoney.com/api/qt/stock/trends2/get?fields1=f1' \
                     '&fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0&secid={}'.format(req_sid)
        else:
            # 获取历史日内数据
            html_m = 'http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1' \
                     '&fields2=f51,f52,f53,f54,f55,f56,f57,f58&ndays={}&iscr=3&secid={}'.format(5,req_sid)
        obj = _parse_url(html_m, bs=False)
        d = json.loads(obj)
        raw_array = [item.split(',') for item in d['data']['trends']]
        minutes= pd.DataFrame(raw_array,columns=['ticker', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'avg'])
        return minutes


class BasicsReader(Reader):

    def __init__(self,engine_path):
        super(BasicsReader,self)._init_db(engine_path)

    def load_stock_basics(self, asset):
        """ 股票基础信息"""
        if asset:
            ins = select([self.symbol_basics]).where(self.symbol_basics.c.sid == asset)
        else:
            ins = select([self.symbol_basics])
        rp = self.conn.execute(ins)
        basics = rp.fetchall()
        return basics

    def load_convertible_basics(self, asset):
        """ 可转债基础信息"""
        ins = select([self.func_basics]).where(self.fund_basics.c.sid == asset)
        rp = self.conn.execute(ins)
        basics = rp.fetchall()
        return basics[0]

    def load_equity_structure(self, asset):
        """
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = self.__getattribute__('symbol_equity')
        ins = select([table.c.代码, table.c.变动日期, table.c.公告日期, cast(table.c.总股本, Numeric(20, 3)).label('总股本'),
                      cast(table.c.流通A股, Numeric(20, 3)), cast(table.c.限售A股, Numeric(20, 3)),
                      cast(table.c.流通B股, Numeric(20, 3)), cast(table.c.流通H股, Numeric(20, 3))]).where(table.c.代码 == asset)
        rp = self.conn.execute(ins)
        equtiy = rp.fetchall()
        return equtiy

    def load_splits_divdend(self,asset):
        """
            1.分红除权
            2.配股
        """
        # connection = engine.raw_connection()
        sql = select([self.symbol_splits]).where(self.symbol_splits.c.sid == asset)
        sql_1 = select([self.symbol_rights]).where(self.symbol_rights.c.sid == asset)
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql,sql_1)
            results_first = cursor.fetchall()
            cursor.nextset()
            results_second = cursor.fetchall()
            cursor.close()
            return results_first , results_second
        finally:
            self.conn.close()

    def load_ashare_holdings(self,sdate,edate,asset):
        """股东持仓变动"""
        table = self.__getattribute__('symbol_share_pct')
        sql = select([table.c.变动截止日,
                      table.c.代码,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.占总流通比例, Numeric(10,5)),
                      cast(table.c.总持仓, Numeric(10,2)),
                      cast(table.c.占总股本比例,Numeric(10,5)),\
                      cast(table.c.总流通股, Numeric(10,2))]).where \
                     (and_(table.c.代码 == asset,table.c.变动截止日.between(sdate,edate)))
        share_pct = self.conn.execute(sql).fetchall()
        pct = pd.DataFrame(share_pct,columns = ['变动截止日','代码','变动股本','占总流通比例','总持仓','占总股本比例','总流通股'])
        return pct

    def load_stock_pledge(self, sid):
        """股票质押率"""
        pledge = ts.to_ts_pledge(sid)
        return pledge


class EventReader(Reader):
    """
        1.大宗交易
        2.解禁数据
        3.市场融资融券
        4.国内GDP
        5.沪港通|深港通标的
    """
    def __init__(self,engine_path):
        super(EventReader,self)._init_db(engine_path)

    def load_market_margin(self, sdate, edate):
        """整个A股市场融资融券余额"""
        table = self.__getattribute__('market_margin')
        sql = select([table.c.交易日期,
                     cast(table.c.融资余额, Integer),
                     cast(table.c.融券余额, Integer),
                     cast(table.c.融资融券总额, Integer),
                     cast(table.c.融资融券差额, Integer)]).where \
                    (table.c.交易日期.between(sdate, edate))
        rp = self.conn.execute(sql).fetchall()
        market_margin = pd.DataFrame(rp, columns=['交易日期', '融资余额', '融券余额', '融资融券总额', '融资融券差额'])
        return market_margin

    def load_ashare_massive(self, sdate, edate):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        massive = pd.DataFrame()
        count = 1
        while True:
            html = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
                   'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&'%count +\
                   'js={"data":(x)}&filter=(Stype=%27EQA%27)'+'(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
            raw = _parse_url(html,bs = False,encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                mass = pd.DataFrame(raw['data'])
                massive = massive.append(mass)
                count = count +1
            else:
                break
        return massive

    def load_ashare_release(self, sdate, edate):
        """
            获取A股解禁数据
        """
        release = pd.DataFrame()
        count = 1
        while True:
            html = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC' \
                   '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p=%d&ps=10&filter=(mkt=)'%count + \
                   '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(sdate,edate) + '&js={"data":(x)}'
            text = _parse_url(html,encoding=None,bs = False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'],item['ltsj'],item['xsglx'],item['zb']] for item in info]
                df = pd.DataFrame(raw,columns = ['代码','解禁时间','类型','解禁占流通市值比例'])
                release = release.append(df)
                count = count + 1
            else:
                break
        return release

    def load_gross_value(self):
        page = 1
        gdp = pd.DataFrame()
        while True:
            html = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=%d' % page
            obj = _parse_url(html)
            raw = obj.findAll('div', {'class': 'Content'})
            text = [t.get_text() for t in raw[1].findAll('td')]
            text = [item.strip() for item in text]
            data = zip(text[::9], text[1::9])
            data = pd.DataFrame(data, columns=['季度', '总值'])
            gdp = gdp.append(data)
            if len(gdp) != len(gdp.drop_duplicates(ignore_index=True)):
                gdp.drop_duplicates(inplace=True, ignore_index=True)
                return gdp
            page = page + 1
        return gdp

    def load_exchange_connection(self, exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 ; flag :1 最新的， 0 为历史的已经踢出的"""
        assets = ts.to_ts_con(exchange, flag)
        return assets