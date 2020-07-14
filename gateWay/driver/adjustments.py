# -*- coding : utf-8 -*-

import sqlalchemy as sa , pandas as pd ,numpy as np
from pandas import Timestamp

from gateWay.tools import unpack_df_to_component_dict
from .db_schema import engine,metadata

ADJUSTMENT_COLUMNS_DTYPE = {
                'sid_bonus':int,
                'sid_transfer':int,
                'bonus':np.float64,
                'right_price':np.float64
                            }


class SQLiteAdjustmentReader(object):
    """
        1 获取所有的分红 配股 数据用于pipeloader
        2.形成特定的格式的dataframe
    """

    # @preprocess(conn=coerce_string_to_conn(require_exists=True))
    def __init__(self):
        self.conn = engine.conncect()

    def __enter__(self):
        return self

    @staticmethod
    def _generate_adjust_dataframe(df):
        df.set_index('sid',inplace = True)
        for col,col_type in ADJUSTMENT_COLUMNS_DTYPE.items():
            try:
                df[col] = df[col].astype(col_type)
            except KeyError:
                pass
        unpack_df = unpack_df_to_component_dict(df)
        return unpack_df

    def _retrieve_divdends(self):
        table = metadata.tables['symbol_divdends']
        sql_dialect = sa.select([table.c.record_date,
                         table.c.sid,
                         sa.cast(table.c.sid_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.sid_transfer,sa.Numeric(5,2)),
                         sa.cast(table.c.bonus,sa.Numeric(5,2))]).\
                        where(table.c.progress.like('实施'))
        rp = self.conn.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(),
                                     columns = ['record_date','sid_bonus','sid_transfer','bonus'])
        adjust_divdends = self._generate_adjust_dataframe(divdends)
        return adjust_divdends

    def _retrieve_rights(self):
        table = metadata.tables['symbol_rights']
        sql = sa.select([table.c.record_date,
                         table.c.sid,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))])
                        # (sa.and_(table.c.sid == asset, table.c.pay_date <= date))
        rp = self.conn.execute(sql)
        rights = pd.DataFrame(rp.fetchall(),
                              columns=['record_date','right_bonus', 'right_price'])
        adjust_rights = self._generate_adjust_dataframe(rights)
        return adjust_rights

    def _load_adjustments_from_sqlite(self,
         should_include_dividends,
         should_include_rights,
        ):
        adjustments = {}
        if should_include_dividends:
            adjustments['divdends'] =  self._retrieve_divdends()
        elif should_include_rights:
            adjustments['rights'] =  self._retrieve_rights()
        else:
            adjustments = None
        return adjustments

    def load_pricing_adjustments(self,
                                 should_include_dividends=True,
                                 should_include_rights=True,
                                 ):
        pricing_divdend = self._load_adjustments_from_sqlite(
                                should_include_dividends,
                                should_include_rights)
        return pricing_divdend

    def load_specific_divdends_for_sid(self,sid,date):
        table = metadata.tables['symbol_divdends']
        sql_dialect = sa.select([table.c.record_date,
                         table.c.sid,
                         sa.cast(table.c.sid_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.sid_transfer,sa.Numeric(5,2)),
                         sa.cast(table.c.bonus,sa.Numeric(5,2))]).where \
                        (sa.and_(table.c.sid == sid,table.c.progress.like('实施'),table.c.pay_date == date))
        rp = self.conn.execute(sql_dialect)
        divdends = pd.DataFrame(rp.fetchall(),
                                     columns = ['record_date','sid_bonus','sid_transfer','bonus'])
        adjust_divdends = self._generate_adjust_dataframe(divdends)
        return adjust_divdends

    def load_specific_rights_for_sid(self,sid,date):
        table = metadata.tables['symbol_rights']
        sql = sa.select([table.c.record_date,
                         table.c.sid,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))]).where\
                        (sa.and_(table.c.sid == sid, table.c.pay_date == date))
        rp = self.conn.execute(sql)
        rights = pd.DataFrame(rp.fetchall(),
                              columns=['record_date','right_bonus', 'right_price'])
        adjust_rights = self._generate_adjust_dataframe(rights)
        return adjust_rights

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()
