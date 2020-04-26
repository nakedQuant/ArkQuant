import sqlalchemy as sa , pandas as pd

class SQLiteAdjustmentReader(object):
    """
    Loads adjustments based on corporate actions from a SQLite database.

    Expects data written in the format output by `SQLiteAdjustmentWriter`.

    Parameters
    ----------
    conn : str or sqlite3.Connection
        Connection from which to load data.

    See Also
    --------
    :class:`zipline.data.adjustments.SQLiteAdjustmentWriter`
    """
    def __init__(self, engine):
        self.conn = engine.conncect()
        self.tables = engine.tables_names()
        self.bar_reader = bar_reader

    def __enter__(self):
        return self

    def _load_divdend_from_sqlite(self,asset,date):
        """stock_divdend cash_divdend"""
        table = self.tables['symbol_splits']
        sql_dialect = sa.select([table.c.ex_date,sa.cast(table.c.payment_sid_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.payment_sid_transfer,sa.Numeric(5,2)),
                         sa.cast(table.c.payment_cash,sa.Numeric(5,2))]).where\
                        (sa.and_(table.c.sid == asset,table.c.progress.like('实施'),table.c.pay_date <= date))
        rp = self.conn.execute(sql_dialect)
        splits_divdend = rp.fetchall()
        formatted_divdend = self._generate_out_dataframe(splits_divdend,
                                     columns = ['pay_date','payment_sid_bonus','payment_sid_transfer','payment_cash'])
        return formatted_divdend

    def _load_issue_from_sqlite(self,asset,date):
        table = self.tables['symbol_rights']
        sql = sa.select([table.c.ex_date,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))]).where\
                        (sa.and_(table.c.sid == asset,table.c.pay_date <= date))
        rp = self.conn(sql)
        issue_divdend = rp.fetchall()
        formatted_issue = self._generate_out_dataframe(issue_divdend,
                                     columns = ['pay_date','right_bonus','right_price'])
        return formatted_issue

    def _generate_out_dataframe(self,raw,columns):
        raw_df = pd.DataFrame(raw,columns = columns)
        raw_df.index = raw_df['ex_date']
        raw_df['preclose'] = map(lambda x : self.bar_reader.get_symbol_price(x,['close']),raw_df.index)
        return raw_df

    def _calculate_divdend_qfq_coef(self,sid,dt):
        """
           hfq --- 后复权 历史价格不变，现价变化
           qfq --- 前复权 现价不变 历史价格变化 --- 前视误差
        """
        raw = self._load_divdend_from_sqlite(sid,dt)
        fq = (1 + (raw['payment_sid_bonus'] +
                   raw['payment_sid_transfer']) / 10) \
             / (1 - raw['payment_cash'] / (10 * raw['preclose']))
        return fq

    def _calculate_issue_qfq_coef(self,sid,dt):
        raw = self._load_issue_from_sqlite(sid,dt)
        issue_price = (raw['right_bonus'] * raw['right_price']
                       + 10 * raw['preclose']) \
                      / (raw['right_bonus'] + 10)
        fq = raw['preclose'] /issue_price
        return fq

    def load_adjustment_for_sid(self,sid,trade_dt):
        """
        Load collection of Adjustment objects from underlying adjustments db.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates for which adjustments are needed.
        assets : pd.Int64Index
            assets for which adjustments are needed.
        should_include_splits : bool
            Whether split adjustments should be included.
        should_include_mergers : bool
            Whether merger adjustments should be included.
        should_include_dividends : bool
            Whether dividend adjustments should be included.
        adjustment_type : str
            Whether price adjustments, volume adjustments, or both, should be
            included in the output.

        Returns
        -------
        adjustments : dict[str -> dict[int -> Adjustment]]
            A dictionary containing price and/or volume adjustment mappings
            from index to adjustment objects to apply at that index.
        """
        fq = self._load_divdend_from_sqlite(sid,trade_dt)
        fq_issue = self._load_issue_from_sqlite(sid,trade_dt)
        fq.append(fq_issue)
        fq.sort_index(ascending= False,inplace = True)
        qfq = 1 / fq.cumprod()
        return qfq

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()