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


__all__ = ['Crawler']


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
