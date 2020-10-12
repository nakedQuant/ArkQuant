# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from sqlalchemy import select, cast, and_, Numeric, Integer
import pandas as pd, sqlalchemy as sa
from toolz import keyfilter, valmap
from gateway.driver.tools import _parse_url, unpack_df_to_component_dict
from gateway.spider.url import ASSET_FUNDAMENTAL_URL
from gateway.driver.bar_reader import BarReader
from gateway.database import engine
from gateway.asset.assets import Equity, Convertible, Fund


__all__ = [
    'MassiveSessionReader',
    'ReleaseSessionReader',
    'HolderSessionReader',
    'OwnershipSessionReader',
    'GrossSessionReader',
    'MarginSessionReader'
]


class MassiveSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_spot_value(self, dt, asset, fields=None):
        table = self.metadata['massive']
        sql = select([cast(table.c.bid_price, Numeric(10, 2)),
                      cast(table.c.discount, Numeric(10, 5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cjeltszb]).where(and_(table.c.declared_date == dt, table.c.sid == asset.sid))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['bid_price', 'discount', 'bid_volume',
                                      'buyer', 'seller', 'cjeltszb'])
        massive_frame = frame.loc[:, fields] if fields else frame
        # index -> 序列号
        massive_frame.drop_duplicates(inplace=True, ignore_index=True)
        return massive_frame

    def load_raw_arrays(self, dts, assets, fields=None):
        sids = [a.sid for a in assets]
        # 获取数据
        table = self.metadata.tables['massive']
        sql = select([table.c.declared_date,
                      table.c.sid,
                      cast(table.c.bid_price, Numeric(10, 2)),
                      cast(table.c.discount, Numeric(10, 5)),
                      cast(table.c.bid_volume, Integer),
                      table.c.buyer,
                      table.c.seller,
                      table.c.cjeltszb]).where(table.c.declared_date.between(dts[0], dts[1]))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['declared_date', 'sid', 'bid_price', 'discount',
                                      'bid_volume', 'buyer', 'seller', 'cjeltszb'])
        frame.set_index('sid', inplace=True)
        frame.drop_duplicates(inplace=True)
        frame_dct = unpack_df_to_component_dict(frame, 'declared_date')
        frame_dct = valmap(lambda x: x.loc[:, fields] if fields else x, frame_dct)
        massive_frame = keyfilter(lambda x: x in sids, frame_dct)
        return massive_frame


class ReleaseSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_spot_value(self, dt, asset, fields=None):
        table = self.metadata.tables['unfreeze']
        sql = select([table.c.release_type,
                      cast(table.c.zb, Numeric(10, 5))]).\
            where(and_(table.c.declared_date == dt, table.c.sid == asset.sid))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['release_type', 'zb'])
        release_frame = frame.loc[:, fields] if fields else frame
        release_frame.drop_duplicates(inplace=True, ignore_index=True)
        return release_frame

    def load_raw_arrays(self, dts, assets, fields=None):
        sids = [a.sid for a in assets]
        table = self.metadata.tables['unfreeze']
        sql = select([table.c.sid,
                      table.c.declared_date,
                      table.c.release_type,
                      cast(table.c.zb, Numeric(10, 5)), ]).\
            where(table.c.declared_date.between(dts[0], dts[1]))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['sid', 'declared_date',
                                      'release_type', 'zb'])
        frame.set_index('sid', inplace=True)
        frame.drop_duplicates(inplace=True)
        frame_dct = unpack_df_to_component_dict(frame, 'declared_date')
        frame_dct = valmap(lambda x: x.loc[:, fields] if fields else x, frame_dct)
        release_frame = keyfilter(lambda x: x in sids, frame_dct)
        return release_frame


class HolderSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_spot_value(self, dt, asset, fields=None):
        """股东持仓变动"""
        table = self.metadata.tables['holder']
        sql = select([table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10, 2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比例, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占总流通比例, Numeric(10, 5))]).\
            where(and_(table.c.declared_date == dt, table.c.sid == asset.sid))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['股东', '方式', '变动股本', '总持仓',
                                      '占总股本比例', '总流通股', '占总流通比例'])
        holder_frame = frame.loc[:, fields] if fields else frame
        holder_frame.drop_duplicates(inplace=True, ignore_index=True)
        return holder_frame

    def load_raw_arrays(self, dts, assets, fields=None):
        """股东持仓变动"""
        sids = [a.sid for a in assets]
        table = self.metadata.tables['holder']
        sql = select([table.c.sid,
                      table.c.declared_date,
                      table.c.股东,
                      table.c.方式,
                      cast(table.c.变动股本, Numeric(10, 2)),
                      cast(table.c.总持仓, Integer),
                      cast(table.c.占总股本比, Numeric(10, 5)),
                      cast(table.c.总流通股, Integer),
                      cast(table.c.占流通比, Numeric(10, 5))]).where(
                    table.c.declared_date.between(dts[0], dts[1]))
        frame = pd.DataFrame(self.engine.execute(sql).fetchall(),
                             columns=['sid', 'declared_date', '股东', '方式', '变动股本',
                                      '总持仓', '占总股本比', '总流通股', '占流通比'])
        frame.set_index('sid', inplace=True)
        frame.drop_duplicates(inplace=True)
        frame_dct = unpack_df_to_component_dict(frame, 'declared_date')
        frame_dct = valmap(lambda x: x.loc[:, fields] if fields else x, frame_dct)
        holder_frame = keyfilter(lambda x: x in sids, frame_dct)
        return holder_frame


class OwnershipSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_spot_value(self, dt, asset, fields=None):
        """
            extra information about asset --- equity structure
            股票的总股本、流通股本，公告日期,变动日期结构
            Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
            Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
            --- 将 -- 变为0
        """
        table = self.metadata.tables['ownership']
        ins = sa.select([table.c.declared_date, table.c.ex_date,
                         sa.cast(table.c.general, sa.Numeric(20, 3)),
                         table.c.float,
                         table.c.manager,
                         table.c.strict]).\
            where(table.c.sid == asset.sid)
        frame = pd.DataFrame(self.engine.execute(ins).fetchall(),
                             columns=['declared_date', 'ex_date', 'general',
                                      'float', 'manager', 'strict'])
        ownership_frame = frame.loc[:, fields] if fields else frame
        ownership_frame.drop_duplicates(inplace=True, ignore_index=True)
        return ownership_frame

    def load_raw_arrays(self, dts, assets, fields=None):
        sids = [a.sid for a in assets]
        table = self.metadata.tables['ownership']
        ins = sa.select([table.c.sid,
                         table.c.declared_date,
                         table.c.ex_date,
                         sa.cast(table.c.general, sa.Numeric(20, 3)),
                         table.c.float, table.c.manager,
                         table.c.strict]).\
            where(table.c.declared_date.between(dts[0], dts[1]))
        frame = pd.DataFrame(self.engine.execute(ins).fetchall(),
                             columns=['sid', 'declared_date', 'ex_date', 'general',
                                      'float', 'manager', 'strict'])
        frame.set_index('sid', inplace=True)
        frame.drop_duplicates(inplace=True)
        frame_dct = unpack_df_to_component_dict(frame, 'declared_date')
        frame_dct = valmap(lambda x: x.loc[:, fields] if fields else x, frame_dct)
        ownership_frame = keyfilter(lambda x: x in sids, frame_dct)
        return ownership_frame


class MarginSessionReader(BarReader):

    def __init__(self):
        self.engine = engine

    @property
    def data_frequency(self):
        return 'daily'

    def get_spot_value(self, dt, asset, fields=None):
        raise NotImplementedError('get_values is deprescated ,use load_raw_arrays method')

    def load_raw_arrays(self, dts, assets, fields=None):
        table = self.metadata.tables['margin']
        ins = sa.select([table.c.declared_date,
                         table.c.rzye,
                         table.c.rzyezb,
                         table.c.rqye]).\
            where(table.c.declared_date.between(dts[0], dts[1]))
        frame = pd.DataFrame(self.engine.execute(ins).fetchall(),
                             columns=['declared_date', 'rzye', 'rzyezb', 'rqye'])
        frame.set_index('declared_date', inplace=True)
        frame.drop_duplicates(inplace=True)
        return frame


class GrossSessionReader(BarReader):

    def __init__(self, url=None):
        self._url = url if url else ASSET_FUNDAMENTAL_URL['gross']

    def get_spot_value(self, dt, asset, field=None):
        raise NotImplementedError('get_values is deprescated by gpd ,use load_raw_arrays method')

    def load_raw_arrays(self, dts, assets, fields=None):
        """获取GDP数据"""
        page = 1
        gross_value = pd.DataFrame()
        while True:
            req_url = self._url % page
            obj = _parse_url(req_url)
            raw = obj.findAll('div', {'class': 'Content'})
            text = [t.get_text() for t in raw[1].findAll('td')]
            text = [item.strip() for item in text]
            data = zip(text[::9], text[1::9])
            data = pd.DataFrame(data, columns=['季度', '总值'])
            gross_value = gross_value.append(data)
            if len(gross_value) != len(gross_value.drop_duplicates(ignore_index=True)):
                gross_value.drop_duplicates(inplace=True, ignore_index=True)
                break
            page = page + 1
        return gross_value


if __name__ == '__main__':

    sessions = ['2020-03-25', '2020-05-01']
    asset = [Equity('600000')]
    massive = MassiveSessionReader()
    m = massive.load_raw_arrays(sessions, asset)
    print('massive', m)
    release = ReleaseSessionReader()
    r = release.load_raw_arrays(sessions, asset)
    print('release', r)
    holder = HolderSessionReader()
    h = holder.load_raw_arrays(sessions, asset)
    print('holder', h)
    ownership = OwnershipSessionReader()
    o = ownership.load_raw_arrays(sessions, asset)
    print('ownership', o)
    marign = MarginSessionReader()
    margin_data = marign.load_raw_arrays(sessions, None)
    print('margin', margin_data)
    gdp = GrossSessionReader()
    g = gdp.load_raw_arrays(sessions, None)
    print('gdp', g)
