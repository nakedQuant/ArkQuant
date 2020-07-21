# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from sqlalchemy import select,cast,and_,Numeric,Integer
import pandas as pd , datetime , sqlalchemy as sa , json
from gateWay.driver.tools import _parse_url
from gateWay.driver._config import ASSET_FUNDAMENTAL_URL
from gateWay.driver.bar_reader import BarReader
from .db_schema import engine


class MassiveSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    def get_value(self, asset, dt):
        table = self.metadata['massive']
        sql = select([cast(table.c.bid_price, Numeric(10,2)),
                      cast(table.c.discount, Numeric(10,5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cleltszb]).where(and_(table.c.trade_dt == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        share_massive = pd.DataFrame(raw,columns = ['bid_price','discount','bid_volume','buyer','seller','cleltszb'])
        return share_massive

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        #获取数据
        table = self.metadata['massive']
        sql = select([table.c.trade_dt,
                      table.c.sid,
                      cast(table.c.bid_price, Numeric(10,2)),
                      cast(table.c.discount, Numeric(10,5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cleltszb]).where(table.c.trade_dt.between(sdate,edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw,columns = ['trade_dt','code','bid_price','discount','bid_volume','buyer','seller','cleltszb'])
        df.set_index('code',inplace= True)
        massive = df.loc[sids]
        return massive


class ReleaseSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    def get_value(self, asset,dt):
        table = self.metadata['release']
        sql = select([cast(table.c.release_type, Numeric(10, 2)),
                      cast(table.c.cjeltszb, Numeric(10, 5)), ]).\
            where(and_(table.c.release_date == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        release = pd.DataFrame(raw, columns=['release_type', 'cjeltszb'])
        return release

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        table = self.metadata['release']
        sql = select([table.c.sid,
                      table.c.release_date,
                      cast(table.c.release_type, Numeric(10, 2)),
                      cast(table.c.cjeltszb, Numeric(10, 5)), ]).where \
            (table.c.release_date.between(sdate, edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw, columns=['sid', 'release_date', 'release_type', 'cjeltszb'])
        df.set_index('sid',inplace= True)
        releases = df.loc[sids]
        return releases


class ShareHolderSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    def get_value(self, asset,dt):
        """股东持仓变动"""
        table = self.metadata['shareholder']
        sql = select([table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).where(and_(table.c.公告日 == dt,table.c.sid == asset.sid))
        raw = self.engine.execute(sql).fetchall()
        share_tracker = pd.DataFrame(raw,columns = ['股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
        return share_tracker

    def load_raw_arrays(self, edate, window,assets):
        sdate = self._window_size_to_dt(edate,window)
        sids = [asset.sid for asset in assets]
        """股东持仓变动"""
        table = self.metadata['shareholder']
        sql = select([table.c.sid,
                      table.c.公告日,
                      table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10,2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).where(
                    table.c.declared_date.between(sdate,edate))
        raw = self.engine.execute(sql).fetchall()
        df = pd.DataFrame(raw,columns = ['sid','公告日','股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
        df.set_index('sid',inplace= True)
        trackers = df.loc[sids]
        return trackers


class StructureSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    def get_value(self,asset, dt):
        """
            extra information about asset --- equity structure
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = self.metadata.tables['equity_structure']
        ins = sa.select([table.c.declared_date, table.c.effective_day,
                         sa.cast(table.c.general, sa.Numeric(20, 3)),
                         sa.cast(table.c.float, sa.Numeric(20, 3)),
                         sa.cast(table.c.strict, sa.Numeric(20, 3))]).where(
            table.c.sid == self.sid)
        rp = self.engine.execute(ins)
        structure = pd.DataFrame(rp.fetchall() ,columns = ['declared_date', 'effective_day', 'general', 'float', 'strict'])
        return structure

    def load_raw_arrays(self, edate, window, assets):
        sdate = self._window_size_to_dt(edate, window)
        sids = [asset.sid for asset in assets]

        table = self.metadata.tables['equity_structure']
        ins = sa.select([table.c.sid,table.c.declared_date,
                         table.c.effective_day,
                         sa.cast(table.c.general, sa.Numeric(20, 3)),
                         sa.cast(table.c.float, sa.Numeric(20, 3)),
                         sa.cast(table.c.strict, sa.Numeric(20, 3))]).where(
            table.c.declared_date.between(sdate, edate))
        rp = self.engine.execute(ins)
        df = pd.DataFrame(rp.fetchall(),columns = ['sid','declared_date', 'effective_day', 'general', 'float', 'strict'])
        df.set_index('sid', inplace=True)
        equity_structures = df.loc[sids]
        return equity_structures


class GrossSessionReader(BarReader):

    def __init__(self,url = None):
        self._url = url if url else ASSET_FUNDAMENTAL_URL['gross']

    def get_value(self, asset, dt, field):
        NotImplementedError ('get_values is deprescated by gpd ,use load_raw_arrays method')

    def load_raw_arrays(self,edate,window):
        """获取GDP数据"""
        page = 1
        gross_value = pd.DataFrame()
        while True:
            req_url = self._url%page
            obj = _parse_url(req_url)
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
        #截取时间戳
        sdate = self._window_size_to_dt(edate,window)
        start_idx = gross_value.index(sdate)
        end_idx = gross_value.index(edate)
        return gross_value.iloc[start_idx:end_idx +1,:]


class MarginSessionReader(BarReader):

    def __init__(self,_url = None):
        self._url = _url if _url else ASSET_FUNDAMENTAL_URL['margin']

    def get_value(self, asset, dt, field):
        raise NotImplementedError('get_values is deprescated ,use load_raw_arrays method')

    def load_raw_arrays(self, edate, window):
        sdate = self._window_size_to_dt(edate,window)
        """获取市场全量融资融券"""
        page = 1
        margin = pd.DataFrame()
        while True:
            req_url = self._url% page
            raw = _parse_url(req_url, bs=False)
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
        #
        start_idx = margin.index(sdate)
        end_idx = margin.index(edate)
        return margin.iloc[start_idx:end_idx +1,:]