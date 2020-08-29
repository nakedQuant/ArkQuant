# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from abc import ABC, abstractmethod
from toolz import valmap
from sqlalchemy import select, func
from gateway.database import metadata, engine


__all__ = ['UserAgent', 'ProxyIp', 'Crawler']


UserAgent = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 '
             '(KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 '
             '(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
             'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 '
             '(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
            ]

ProxyIp = ['http://0825fq11t1m612:0825fq11t1m612@117.40.5.82:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.7.122:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.5.35:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.5.128:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.6.202:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.5.94:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.5.28:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.7.9:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.7.100:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.7.200:65000',
           'http://0825fq11t1m612:0825fq11t1m612@117.40.7.165:65000', None]


class Crawler(ABC):

    @property
    def metadata(self):
        return metadata

    @property
    def engine(self):
        return engine

    def _retrieve_assets_from_sqlite(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.sid, table.c.asset_type])
        rp = self.engine.execute(ins)
        assets = pd.DataFrame(rp.fetchall(), columns=['sid', 'asset_type'])
        assets.set_index('sid', inplace=True)
        grp = assets.groupby('asset_type').groups
        mapping = valmap(lambda x: list(x), grp)
        return mapping

    def _retrieve_deadline_from_sqlite(self, tbl):
        table = self.metadata.tables[tbl]
        ins = select([func.max(table.c.declared_date)])
        rp = self.engine.execute(ins)
        deadline = rp.scalar()
        return deadline

    @abstractmethod
    def writer(self, *args):
        """
            intend to spider data from online
        :return:
        """
        raise NotImplementedError()
