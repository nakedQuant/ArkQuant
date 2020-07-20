# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import create_engine,MetaData,select,cast ,Numeric ,Integer
import pandas as pd ,json ,datetime
from abc import ABC
from functools import lru_cache
from toolz import groupby,keyfilter
from db_schema import ENGINE_PTH,metadata
from gateWay.tools import  _parse_url
from gateWay.spider._config import EXTRA_KLINE_REQUEST_URL,ASSETS_BUNDLES_URL
from gateWay.driver.trading_calendar import  TradingCalendar
from gateWay.tools import unpack_df_to_component_dict


__all__ = ['BarReader','EventLoader','ExtraLoader']


class Reader(ABC):

    trading_days = TradingCalendar()

    def _init_db(self):
        engine = create_engine(ENGINE_PTH)
        self.conn = engine.connect.execution_options(
                    isolation_level="READ UNCOMMITTED")
        metadata = MetaData(bind=engine)
        for table_name in engine.table_names():
            setattr(self,table_name,metadata.tables[table_name])


class BarReader(Reader):

    def __init__(self):
        super(BarReader,self)._init_db()

    def _get_kline_from_sqlite(self,start_date,end_date,tbl):
        tbl = metadata[tbl]
        if tbl == 'symbol_price':
            orm = select([tbl.c.trade_dt,tbl.c.sid,
                              cast(tbl.c.open,Numeric(10,2)).label('open'),
                              cast(tbl.c.close,Numeric(12,2)).label('close'),
                              cast(tbl.c.high,Numeric(10,2)).label('high'),
                              cast(tbl.c.low,Numeric(10,3)).label('low'),
                              cast(tbl.c.volume,Numeric(15,0)).label('volume'),
                              cast(tbl.c.amount,Numeric(15,2)).label('amount'),
                              cast(tbl.c.pct,Numeric(5,2)).label('pct')]).\
                    where(tbl.c.trade_dt.between(start_date, end_date))
            rp = self.conn.execute(orm)
            try:
                arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume, r.pct] for r in rp.fetchall()]
                kline = pd.DataFrame(arrays,
                                     columns=['trade_dt', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount',
                                              'pct'])
            except Exception as e:
                kline = None
        else:
            orm = select([tbl.c.trade_dt, tbl.c.sid,
                                  cast(tbl.c.open, Numeric(10, 2)).label('open'),
                                  cast(tbl.c.close, Numeric(12, 2)).label('close'),
                                  cast(tbl.c.high, Numeric(10, 2)).label('high'),
                                  cast(tbl.c.low, Numeric(10, 3)).label('low'),
                                  cast(tbl.c.volume, Numeric(15, 0)).label('volume'),
                                  cast(tbl.c.amount, Numeric(15, 2)).label('amount')]). \
                where(tbl.c.trade_dt.between(start_date, end_date))
            rp = self.conn.execute(orm)
            try:
                arrays = [[r.trade_dt, r.code, r.open, r.close, r.high, r.low, r.volume, r.amount] for r in rp.fetchall()]
                kline = pd.DataFrame(arrays,
                                     columns=['trade_dt', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount'])
            except Exception as e:
                kline = None
        return kline

    @lru_cache(maxsize=8)
    def _transform_raw_arrays(self, edate,window,fields,tbl):
        #处理 fields
        fields_copy = fields.copy()
        fields_copy.append('trade_dt')
        #处理时间
        sdate = self.trading_days.shift_calendar(edate, window)
        #获取数据 ---  dataframe
        raw = self._get_kline_from_sqlite(sdate, edate,tbl)
        raw.set_index('code',inplace= True)
        #基于fields 获取数据
        kline = raw.loc[:, fields_copy]
        unpack_kline = unpack_df_to_component_dict(kline)
        return unpack_kline

    def load_asset_kline(self,date,window,fields,**kwargs):
        """kwargs --- _type(获取对应类别的所有数据）, assets --- 获取对应的标的的数据，不同类型"""
        try:
            category = kwargs['category']
            kline_dict = self._transform_raw_arrays(date,window,fields,'%s_price'%category)
        except KeyError :
            kline_dict = dict()
            #assets 按照标的类别进行分类
            group = groupby(lambda x : x._type,kwargs['assets'])
            for _type,assets in group.items():
                unpack = self._transform_raw_arrays(date, window, fields, '%s_price' % _type)
                kline_dict.update(keyfilter(lambda x: x in assets,unpack))
        return kline_dict

    @classmethod
    def load_hk_kline(cls,sid,edate,window, mode='qfq'):
        """
            获取港股Kline , 针对于同时在A股上市的 , AH
            load_daily_hSymbol('00168', '2011-01-01', '2012-01-06')
            'us' + '.' + code
        """
        sdate = cls.trading_days.shift_calendar(edate,window)
        attached_symbol = 'uk' + sid
        url = ASSETS_BUNDLES_URL['hongkong']%( attached_symbol, sdate, edate, mode)
        raw = _parse_url(url, bs=False, encoding=None)
        raw = json.loads(raw)
        data = raw['data']
        if data and len(data):
            daily = [item[:6] for item in data[attached_symbol]['day']]
            df = pd.DataFrame(daily,columns=['trade_dt','open','close','high','low','volume'])
            df.loc[:,'hSid'] = sid
        return df

    @staticmethod
    def load_minutes_kline(sid,lag):
        req_sid = '0.' + sid if sid.startswith('6') else '1.' + sid
        url = ASSETS_BUNDLES_URL['minute'].format(req_sid) if lag == 0 \
                else ASSETS_BUNDLES_URL['minute_his'].format(lag,req_sid)
        obj = _parse_url(url, bs=False)
        d = json.loads(obj)
        raw_array = [item.split(',') for item in d['data']['trends']]
        minutes= pd.DataFrame(raw_array,columns=['ticker', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'avg'])
        return minutes


class EventLoader(Reader):
    """
        1.大宗交易
        2.解禁数据
        3.股东持仓变动
    """
    def __init__(self):
        super(EventLoader,self)._init_db()

    def load_holder_kline(self,edate,window):
        sdate = self.trading_days.shift_calendar(edate,window)
        """股东持仓变动"""
        table = metadata['shareholder']
        sql = select([table.c.sid,
                      table.c.公告日,
                      table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).where(
                    table.c.公告日.between(sdate,edate))
        raw = self.conn.execute(sql).fetchall()
        share_change = pd.DataFrame(raw,columns = ['sid','公告日','股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
        return share_change

    def load_massive_kline(self,edate,window):
        sdate = self.trading_days.shift_calendar(edate,window)
        table = metadata['massive']
        sql = select([table.c.trade_dt,
                      table.c.sid,
                      cast(table.c.bid_price, Numeric(10,2)),
                      cast(table.c.discount, Numeric(10,5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cleltszb]).where(table.c.trade_dt.between(sdate,edate))
        raw = self.conn.execute(sql).fetchall()
        share_massive = pd.DataFrame(raw,columns = ['trade_dt','sid','bid_price','discount','bid_volume','buyer','seller','cleltszb'])
        return share_massive

    def load_release_kline(self,edate,window):
        sdate = self.trading_days.shift_calendar(edate,window)
        table = metadata['release']
        sql = select([table.c.sid,
                      table.c.release_date,
                      cast(table.c.release_type, Numeric(10,2)),
                      cast(table.c.cjeltszb, Numeric(10,5)),]).where \
                     (table.c.release_date.between(sdate,edate))
        raw = self.conn.execute(sql).fetchall()
        share_release = pd.DataFrame(raw,columns = ['sid','release_date','release_type','cjeltszb'])
        return share_release


class ExtraordinaryLoader(Reader):

    @classmethod
    def load_gdp_kline(cls):
        """获取GDP数据"""
        page = 1
        gross_value = pd.DataFrame()
        while True:
            url = EXTRA_KLINE_REQUEST_URL['GDP']%page
            obj = _parse_url(url)
            raw = obj.findAll('div', {'class': 'Content'})
            text = [t.get_text() for t in raw[1].findAll('td')]
            text = [item.strip() for item in text]
            data = zip(text[::9], text[1::9])
            data = pd.DataFrame(data, columns=['季度', '总值'])
            gross_value = gross_value.append(data)
            if len(gross_value) != len(gross_value.drop_duplicates(ignore_index=True)):
                gross_value.drop_duplicates(inplace=True, ignore_index=True)
                return gross_value
            page = page + 1
        return gross_value

    @classmethod
    def load_margin_kline(cls):
        """获取市场全量融资融券"""
        page = 1
        margin = pd.DataFrame()
        while True:
            url = EXTRA_KLINE_REQUEST_URL['margin']
            raw = _parse_url(url, bs=False)
            raw = json.loads(raw)
            raw = [
                [item['dim_date'], item['rzye'], item['rqye'], item['rzrqye'], item['rzrqyecz'], item['new'],
                 item['zdf']]
                for item in raw['data']]
            data = pd.DataFrame(raw, columns=['trade_dt', 'rzye', 'rqye', 'rzrqze', 'rzrqce', 'hs300', 'pct'])
            data.loc[:, 'trade_dt'] = [datetime.datetime.fromtimestamp(dt / 1000) for dt in data['trade_dt']]
            data.loc[:, 'trade_dt'] = [datetime.datetime.strftime(t, '%Y-%m-%d') for t in data['trade_dt']]
            if len(data) == 0:
                break
            margin = margin.append(data)
            page = page + 1
        margin.set_index('trade_dt', inplace=True)
        return margin

    @classmethod
    def  load_blackSwat_kline(cls,dt):
        """获取某一时点的黑天鹅数据 --- 标的 类型 """
        raise NotImplementedError