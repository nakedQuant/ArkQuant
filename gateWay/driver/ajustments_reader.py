import sqlalchemy as sa

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
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        return self.conn.close()

    def attach_ex_date(self,raw):
        pass

    def _get_sid_of_preclose(self,sid,date):


    def _get_divdend_from_sqlite(self,asset,ex_date):
        """stock_divdend cash_divdend"""
        table = self.tables['symbol_splits']
        sql_dialect = sa.select([table.c.pay_date,sa.cast(table.c.payment_sid_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.payment_sid_transfer,sa.Numeric(5,2)),
                         sa.cast(table.c.payment_cash,sa.Numeric(5,2))]).where\
                        (sa.and_(table.c.sid == asset,table.c.progress.like('实施'),table.c.pay_date <= ex_date))
        rp = self.conn.execute(sql_dialect)
        splits_divdend = rp.fetchall()
        return splits_divdend

    def _get_issue_from_sqlite(self,asset,ex_date):
        table = self.tables['symbol_rights']
        sql = sa.select([table.c.pay_date,
                         sa.cast(table.c.right_bonus,sa.Numeric(5,2)),
                         sa.cast(table.c.right_price,sa.Numeric(5,2))]).where\
                        (sa.and_(table.c.sid == asset,table.c.pay_date <= ex_date))
        rp = self.conn(sql)
        issue_divdend = rp.fetchall()
        return issue_divdend

    def _calculate_divdend_qfq_coef(self,sid):
        """
           hfq --- 后复权 历史价格不变，现价变化
           qfq --- 前复权 现价不变 历史价格变化 --- 前视误差
        """
        raw = self._get_divdend_from_sqlite(sid)
        raw['preclose'] = map(lambda x : self._get_preclose(x,sid),raw['pay_date'])
        qfq_s = ( 1 + (raw['payment_sid_bonus'] + raw['payment_sid_transfer']) / 10 )
        qfq_d = 1 - raw['payment_cash'] / (10 * preclose)
        coef = raw['coef'].cumprod()
        coef.fillna(method='ffill', inplace=True)
        coef.fillna(method='bfill', inplace=True)

    def _calculate_issue_qfq_coef(self,sid):
        raw = self._get_issue_from_sqlite(sid)
        issue_close = (raw['right_bonus'] * raw['right_price'] + 10 * preclose) / preclose
        rate = issue_close / preclose
        return rate

    def load_adjustment_for_sid(self,
                         dates,
                         sid,
                         should_include_splits,
                         should_include_mergers,
                         should_include_dividends,
                         adjustment_type):
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
        divdends = self._get_divdend_from_sqlite(sid,date)
        issues = self._get_issue_from_sqlite(sid,date)
        close = self._get_close()


    def load_pricing_adjustments(self, columns, dates, assets):
        if 'volume' not in set(columns):
            adjustment_type = 'price'
        elif len(set(columns)) == 1:
            adjustment_type = 'volume'
        else:
            adjustment_type = 'all'

        adjustments = self.load_adjustments(
            dates,
            assets,
            should_include_splits=True,
            should_include_mergers=True,
            should_include_dividends=True,
            adjustment_type=adjustment_type,
        )
        price_adjustments = adjustments.get('price')
        volume_adjustments = adjustments.get('volume')

        return [
            volume_adjustments if column == 'volume'
            else price_adjustments
            for column in columns
        ]

    def get_df_from_table(self, table_name, convert_dates=False):

        result = pd.read_sql(
            'select * from "{}"'.format(table_name),
            self.conn,
            index_col='index',
            **kwargs
        ).rename_axis(None)

        if not len(result):
            dtypes = self._df_dtypes(table_name, convert_dates)
            return empty_dataframe(*keysorted(dtypes))

        return result

    def _df_dtypes(self, table_name, convert_dates):
        """Get dtypes to use when unpacking sqlite tables as dataframes.
        """
        out = self._raw_table_dtypes[table_name]
        if convert_dates:
            out = out.copy()
            for date_column in self._datetime_int_cols[table_name]:
                out[date_column] = datetime64ns_dtype

        return out
