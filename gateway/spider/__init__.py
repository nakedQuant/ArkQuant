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
        assets = pd.DataFrame(rp.fetchall(), columns=['sid', 'category'])
        # add asset_type
        assets.loc[:, 'asset_type'] = assets['category'].apply(lambda x:
                                                               'fund' if x not in ['equity', 'convertible'] else x)
        assets.set_index('sid', inplace=True)
        grp = assets.groupby('asset_type').groups
        mapping = valmap(lambda x: list(x), grp)
        return mapping

    # declared_date --- splits rights and ownership
    def _retrieve_deadlines_from_sqlite(self, tbl, date_type='declared_date'):
        table = self.metadata.tables[tbl]
        try:
            if date_type == 'declared_date':
                ins = select([func.max(table.c.declared_date), table.c.sid])
            else:
                ins = select([func.max(table.c.ex_date), table.c.sid])
            ins = ins.group_by(table.c.sid)
            rp = self.engine.execute(ins)
            deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
            deadlines.set_index('sid', inplace=True)
            deadlines = deadlines.iloc[:, 0]
        except AttributeError:
            ins = select([func.max(table.c.declared_date)])
            rp = self.engine.execute(ins)
            deadlines = rp.scalar()
        return deadlines

    # latest date for equity_price convertible_price fund_price
    def _retrieve_latest_from_sqlite(self, tbl):
        table = self.metadata.tables[tbl]
        ins = select([func.max(table.c.trade_dt), table.c.sid])
        ins = ins.group_by(table.c.sid)
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(), columns=['trade_dt', 'sid'])
        deadlines.set_index('sid', inplace=True)
        deadlines = deadlines.iloc[:, 0]
        return deadlines

    @abstractmethod
    def _writer_internal(self, *args):
        """
            internal method for writer
        """
        raise NotImplementedError()

    def writer(self, *args):
        """
            intend to api _writer_internal
        """
        raise NotImplementedError()


__all__ = ['Crawler']
