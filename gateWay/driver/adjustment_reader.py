# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np,pandas as pd,sqlalchemy as sa
from sqlalchemy import and_
from gateWay.driver.db_schema import  metadata
from gateWay.driver.tools import  unpack_df_to_component_dict
from calendar.trading_calendar import  TradingCalendar

ADJUSTMENT_COLUMNS_DTYPE = {
                'sid_bonus':int,
                'sid_transfer':int,
                'bonus':np.float64,
                'right_price':np.float64
                            }

trading_calendar = TradingCalendar()


class SQLiteAdjustmentReader(object):
    """
        1 获取所有的分红 配股 数据用于pipeloader
        2.形成特定的格式的dataframe
    """
    adjustment_tables = frozenset(['equity_divdends','equity_rights'])

    # @preprocess(conn=coerce_string_to_conn(require_exists=True))
    def __init__(self,engine):
        self.engine = engine
        for tbl in self.adjustment_tables:
            setattr(self, tbl, metadata.tables[tbl])

    def __enter__(self):
        return self

    @property
    def _calendar(self):
        return trading_calendar

    def _window_offset_dts(self,date,window):
        sessions = self._calendar.sessions_in_range(date,window)
        return sessions

    def _get_divdends_with_ex_date(self,session):
        sql_dialect = sa.select([self.equity_divdend.c.sid,
                                self.equity_divdend.c.ex_date,
                                sa.cast(self.equity_divdend.c.sid_bonus,sa.Numeric(5,2)),
                                sa.cast(self.equity_divdend.c.sid_transfer,sa.Numeric(5,2)),
                                sa.cast(self.equity_divdend.c.bonus,sa.Numeric(5,2))]).\
                                where(and_(self.equity_divdend.c.pay_date.between(session[0],session[1]),
                                           self.equity_divdend.c.progress.like('实施')))
        rp = self.engine.execute(sql_dialect)
        divdends = pd.DataFrame(
                                rp.fetchall(),
                                columns = ['code','ex_date','sid_bonus',
                                               'sid_transfer','bonus']
                                )
        adjust_divdends = self._generate_dict_from_dataframe(divdends)
        return adjust_divdends

    def _get_rights_with_ex_date(self,session):
        sql = sa.select([self.equity_rights.c.sid,
                         self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(self.equity_rights.c.right_price,sa.Numeric(5,2))]).\
                        where(sa.self.equity_rights.c.pay_date.between(session[0],session[1]))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(
                            rp.fetchall(),
                            columns=['code','ex_date',
                                     'right_bonus','right_price']
                            )
        adjust_rights = self._generate_dict_from_dataframe(rights)
        return adjust_rights

    @staticmethod
    def _generate_dict_from_dataframe(df):
        df.set_index('sid',inplace = True)
        for col,col_type in ADJUSTMENT_COLUMNS_DTYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                raise TypeError('%s cannot mutate into %s'%(col,col_type))
        #asset : splits or divdend or rights
        unpack_df = unpack_df_to_component_dict(df)
        return unpack_df

    def _load_adjustments_from_sqlite(self,sessions,
         should_include_dividends,
         should_include_rights,
        ):
        adjustments = {}
        if should_include_dividends:
            adjustments['divdends'] = self._get_divdends_with_ex_date(sessions)
        elif should_include_rights:
            adjustments['rights'] = self._get_rights_with_ex_date(sessions)
        else:
            adjustments = None
        return adjustments

    def load_pricing_adjustments(self,date,window,
                                 should_include_dividends=True,
                                 should_include_rights=True,
                                 ):
        sessions = self._window_offset_dts(date,window)
        pricing_adjustments = self._load_adjustments_from_sqlite(sessions,
                                should_include_dividends,
                                should_include_rights)
        return pricing_adjustments

    def load_divdends_for_sid(self,sid,date):
        sql_dialect = sa.select([self.equity_divdends.c.ex_date,
                         sa.cast(self.equity_divdends.c.sid_bonus,sa.Numeric(5,2)),
                         sa.cast(self.equity_divdends.c.sid_transfer,sa.Numeric(5,2)),
                         sa.cast(self.equity_divdends.c.bonus,sa.Numeric(5,2))]).\
                        where(sa.and_(self.equity_divdends.c.sid == sid,
                                      self.equity_divdends.c.progress.like('实施'),
                                      self.equity_divdends.c.ex_date == date))
        rp = self.conn.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(),
                                     columns = ['ex_date','sid_bonus','sid_transfer','bonus'])
        return divdends

    def load_rights_for_sid(self,sid,date):
        sql = sa.select([self.equity_rights.c.ex_date,
                         sa.cast(self.equity_rights.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(self.equity_rights.c.right_price,sa.Numeric(5,2))]).\
                        where(sa.and_(self.equity_rights.c.sid == sid,
                                      self.equity_rights.c.ex_date == date))
        rp = self.engine.execute(sql)
        rights = pd.DataFrame(rp.fetchall(),
                              columns=['ex_date','right_bonus', 'right_price'])
        return rights

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()