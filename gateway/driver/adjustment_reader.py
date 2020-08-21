# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_, MetaData
from gateway.database import engine
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
    adjustment_tables = frozenset(['equity_divdends', 'equity_rights'])

    def __init__(self):
        self.engine = engine
        metadata = MetaData(bind=engine)
        for tbl in self.adjustment_tables:
            setattr(self, tbl, metadata.tables[tbl])

    def __enter__(self):
        return self

    @staticmethod
    def _resample_frame_to_dict(df):
        df.set_index('sid', inplace=True)
        for col, col_type in ADJUSTMENT_COLUMNS_TYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                pass
            except TypeError:
                raise TypeError('%s cannot mutate into %s'%(col, col_type))
        unpack_df = unpack_df_to_component_dict(df)
        return unpack_df

    def _get_divdends_with_ex_date(self, sessions):
        sdate, edate = sessions
        sql_dialect = sa.select([self.equity_divdend.c.sid,
                                self.equity_divdend.c.ex_date,
                                sa.cast(self.equity_divdend.c.sid_bonus, sa.Numeric(5, 2)),
                                sa.cast(self.equity_divdend.c.sid_transfer, sa.Numeric(5, 2)),
                                sa.cast(self.equity_divdend.c.bonus, sa.Numeric(5, 2))]).\
            where(and_(self.equity_divdend.c.pay_date.between(sdate, edate), self.equity_divdend.c.progress.like('实施')))
        rp = self.engine.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(), columns=['sid', 'ex_date', 'sid_bonus',
                                                        'sid_transfer', 'bonus'])
        adjust_divdends = self._resample_frame_to_dict(divdends)
        return adjust_divdends

    def _get_rights_with_ex_date(self, sessions):
        sdate, edate = sessions
        sql = sa.select([self.equity_rights.c.sid,
                         self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.right_bonus,sa.Numeric(5, 2)),
                         sa.cast(self.equity_rights.c.right_price,sa.Numeric(5, 2))]).\
            where(sa.self.equity_rights.c.pay_date.between(sdate, edate))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(rp.fetchall(), columns=['code', 'ex_date',
                                                      'right_bonus', 'right_price'])
        adjust_rights = self._resample_frame_to_dict(rights)
        return adjust_rights

    def _load_adjustments_from_sqlite(self,
                                      sessions,
                                      should_include_dividends,
                                      should_include_rights):
        adjustments = {}
        if should_include_dividends:
            adjustments['divdends'] = self._get_divdends_with_ex_date(sessions)
        elif should_include_rights:
            adjustments['rights'] = self._get_rights_with_ex_date(sessions)
        else:
            raise ValueError('must include divdends or rights')
        return adjustments

    def load_pricing_adjustments(self, sessions,
                                 should_include_dividends=True,
                                 should_include_rights=True,
                                 ):
        pricing_adjustments = self._load_adjustments_from_sqlite(
                                sessions,
                                should_include_dividends,
                                should_include_rights)
        return pricing_adjustments

    def load_divdends_for_sid(self, sid, date):
        sql_dialect = sa.select([self.equity_divdends.c.ex_date,
                                 sa.cast(self.equity_divdends.c.sid_bonus, sa.Numeric(5, 2)),
                                 sa.cast(self.equity_divdends.c.sid_transfer, sa.Numeric(5, 2)),
                                 sa.cast(self.equity_divdends.c.bonus, sa.Numeric(5, 2))]).\
                                where(sa.and_(self.equity_divdends.c.sid == sid,
                                      self.equity_divdends.c.progress.like('实施'),
                                      self.equity_divdends.c.pay_date == date))
        rp = self.conn.execute(sql_dialect)
        dividends = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'sid_bonus',
                                                         'sid_transfer', 'bonus'])
        return dividends

    def load_rights_for_sid(self, sid, date):
        sql = sa.select([self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.right_bonus,sa.Numeric(5, 2)),
                         sa.cast(self.equity_rights.c.right_price,sa.Numeric(5, 2))]).\
                        where(sa.and_(self.equity_rights.c.sid == sid,
                                      self.equity_rights.c.pay_date == date))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'right_bonus', 'right_price'])
        return rights

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()
