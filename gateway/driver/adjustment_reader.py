# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_
from gateway.database import engine, metadata
from gateway.driver.tools import unpack_df_to_component_dict


ADJUSTMENT_COLUMNS_TYPE = {
                'sid_bonus': int,
                'sid_transfer': int,
                'bonus': np.float64,
                'right_bonus': int,
                'right_price': np.float64
                        }


class SQLiteAdjustmentReader(object):
    """
        1 获取所有的分红 配股 数据用于pipeloader
        2.形成特定的格式的dataframe
    """
    adjustment_tables = frozenset(['equity_splits', 'equity_rights'])

    def __init__(self):
        self.engine = engine
        metadata.reflect(bind=engine)
        for tbl in self.adjustment_tables:
            setattr(self, tbl, metadata.tables[tbl])

    def __enter__(self):
        return self

    @staticmethod
    def _adjust_frame_type(df):
        for col, col_type in ADJUSTMENT_COLUMNS_TYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                pass
            except TypeError:
                raise TypeError('%s cannot mutate into %s' % (col, col_type))
        return df

    def _get_divdends_with_ex_date(self, sessions):
        sdate, edate = sessions
        sql_dialect = sa.select([self.equity_splits.c.sid,
                                self.equity_splits.c.ex_date,
                                sa.cast(self.equity_splits.c.sid_bonus, sa.Numeric(5, 2)),
                                sa.cast(self.equity_splits.c.sid_transfer, sa.Numeric(5, 2)),
                                sa.cast(self.equity_splits.c.bonus, sa.Numeric(5, 2))]).\
            where(and_(self.equity_splits.c.pay_date.between(sdate, edate), self.equity_splits.c.progress.like('实施')))
        rp = self.engine.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(), columns=['sid', 'ex_date', 'sid_bonus',
                                                        'sid_transfer', 'bonus'])
        divdends.set_index('sid', inplace=True)
        adjust_divdends = self._adjust_frame_type(divdends)
        unpack_divdends = unpack_df_to_component_dict(adjust_divdends, 'ex_date')
        return unpack_divdends

    def _get_rights_with_ex_date(self, sessions):
        sdate, edate = sessions
        sql = sa.select([self.equity_rights.c.sid,
                         self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.rights_bonus, sa.Numeric(5, 2)),
                         sa.cast(self.equity_rights.c.rights_price, sa.Numeric(5, 2))]).\
            where(self.equity_rights.c.pay_date.between(sdate, edate))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(rp.fetchall(), columns=['sid', 'ex_date',
                                                      'rights_bonus', 'rights_price'])
        rights.set_index('sid', inplace=True)
        adjust_rights = self._adjust_frame_type(rights)
        unpack_rights = unpack_df_to_component_dict(adjust_rights, 'ex_date')
        return unpack_rights

    def _load_adjustments_from_sqlite(self, sessions):
        adjustments = {}
        # retrieve from mysql
        adjustments['divdends'] = self._get_divdends_with_ex_date(sessions)
        adjustments['rights'] = self._get_rights_with_ex_date(sessions)
        return adjustments

    def load_pricing_adjustments(self, sessions):
        pricing_adjustments = self._load_adjustments_from_sqlite(sessions)
        return pricing_adjustments

    def load_divdends_for_sid(self, sid, date):
        sql_dialect = sa.select([self.equity_splits.c.ex_date,
                                 sa.cast(self.equity_splits.c.sid_bonus, sa.Numeric(5, 2)),
                                 sa.cast(self.equity_splits.c.sid_transfer, sa.Numeric(5, 2)),
                                 sa.cast(self.equity_splits.c.bonus, sa.Numeric(5, 2))]).\
                                where(sa.and_(self.equity_splits.c.sid == sid,
                                      self.equity_splits.c.progress.like('实施'),
                                      self.equity_splits.c.pay_date == date))
        rp = self.engine.execute(sql_dialect)
        dividends = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'sid_bonus',
                                                         'sid_transfer', 'bonus'])
        adjust_divdends = self._adjust_frame_type(dividends)
        return adjust_divdends

    def load_rights_for_sid(self, sid, date):
        sql = sa.select([self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.rights_bonus, sa.Numeric(5, 2)),
                         sa.cast(self.equity_rights.c.rights_price, sa.Numeric(5, 2))]).\
                        where(sa.and_(self.equity_rights.c.sid == sid,
                                      self.equity_rights.c.pay_date == date))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'right_bonus', 'right_price'])
        adjust_rights = self._adjust_frame_type(rights)
        return adjust_rights

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.engine.dispose()


if __name__ == '__main__':

    reader = SQLiteAdjustmentReader()
    adjustments = reader.load_pricing_adjustments(['2018-01-01', '2020-09-01'])
    print('adjustments.keys', adjustments.keys())
    # symbol_divdends = reader.load_divdends_for_sid('600061', '20200401')
    # print('divdends', symbol_divdends)
    # symbol_rights = reader.load_rights_for_sid('600061', '20200401')
    # print('rights', symbol_rights)
