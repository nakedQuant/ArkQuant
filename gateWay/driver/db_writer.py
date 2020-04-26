from sqlalchemy import MetaData,create_engine
from weakref import WeakValueDictionary

asset_db_table_names = frozenset({
    'symbol_naive_price',
    'dual_symbol_price'
    'bond_price',
    'index_price',
    'fund_price',
    'symbol_equity_basics',
    'bond_basics',
    'symbol_splits',
    'symbol_issue',
    'symbol_mcap',
    'symbol_massive',
    'market_margin',
    'version_info',
})

class DBCommon(object):
    """
        实现基本共功能 execute fetchall
    """



class DBWriter(object):

    engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/test',
                           pool_size=50, max_overflow=100, pool_timeout=-1)


    _cache = WeakValueDictionary()

    def __new__(cls,engine_path):
        try:
            return cls._cache[engine_path]
        except KeyError:
            class_instance = cls._cache[engine_path] = \
                             super(DBWriter,cls).__new__()._init_db(engine_path)
            return class_instance

    def _init_db(self,engine_path):
        self.engine = create_engine(engine_path)
        # if len(self.engine.table_names())
        self.metadata = MetaData()
        self.metadata.create_all(bind = engine)

    def reset_db(self):
        """
            重置数据库 ———— 清空数据
        """
        self.metadata.reflect(bind = self.engine)
        for tbl in self.metadata.sorted_tables:
            self.engine.execute(tbl.delete())

    def writer(self,tbl,df):
        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            self.init_db(conn)
        self._write_df_to_table(tbl,df)

    def _write_df_to_table(self, tbl, df,chunk_size):
        df.to_sql(
            tbl,
            self.engine,
            index=True,
            index_label=None,
            if_exists='append',
            chunksize=chunk_size,
        )

    def set_isolation_level(self):
        connection = self.engine.connect()
        connection = connection.execution_options(
            isolation_level="READ COMMITTED"
        )


    def check_version_info(conn, version_table, expected_version):
        """
        Checks for a version value in the version table.

        Parameters
        ----------
        conn : sa.Connection
            The connection to use to perform the check.
        version_table : sa.Table
            The version table of the asset database
        expected_version : int
            The expected version of the asset database

        Raises
        ------
        AssetDBVersionError
            If the version is in the table and not equal to ASSET_DB_VERSION.
        """

        # Read the version out of the table
        version_from_table = conn.execute(
            sa.select((version_table.c.version,)),
        ).scalar()

        # A db without a version is considered v0
        if version_from_table is None:
            version_from_table = 0

        # Raise an error if the versions do not match
        if (version_from_table != expected_version):
            raise TypeError('version ---')

class SQLiteAdjustmentWriter(object):
    """
    Writer for data to be read by SQLiteAdjustmentReader

    Parameters
    ----------
    conn_or_path : str or sqlite3.Connection
        A handle to the target sqlite database.
    equity_daily_bar_reader : SessionBarReader
        Daily bar reader to use for dividend writes.
    overwrite : bool, optional, default=False
        If True and conn_or_path is a string, remove any existing files at the
        given path before connecting.

    See Also
    --------
    zipline.data.adjustments.SQLiteAdjustmentReader
    """

    def __init__(self, conn_or_path, equity_daily_bar_reader, overwrite=False):
        if isinstance(conn_or_path, sqlite3.Connection):
            self.conn = conn_or_path
        elif isinstance(conn_or_path, six.string_types):
            if overwrite:
                try:
                    remove(conn_or_path)
                except OSError as e:
                    if e.errno != ENOENT:
                        raise
            self.conn = sqlite3.connect(conn_or_path)
            self.uri = conn_or_path
        else:
            raise TypeError("Unknown connection type %s" % type(conn_or_path))

        self._equity_daily_bar_reader = equity_daily_bar_reader

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self.conn.close()

    def _write(self, tablename, expected_dtypes, frame):
        if frame is None or frame.empty:
            # keeping the dtypes correct for empty frames is not easy
            frame = pd.DataFrame(
                np.array([], dtype=list(expected_dtypes.items())),
            )
        else:
            if frozenset(frame.columns) != frozenset(expected_dtypes):
                raise ValueError(
                    "Unexpected frame columns:\n"
                    "Expected Columns: %s\n"
                    "Received Columns: %s" % (
                        set(expected_dtypes),
                        frame.columns.tolist(),
                    )
                )

            actual_dtypes = frame.dtypes
            for colname, expected in six.iteritems(expected_dtypes):
                actual = actual_dtypes[colname]
                if not np.issubdtype(actual, expected):
                    raise TypeError(
                        "Expected data of type {expected} for column"
                        " '{colname}', but got '{actual}'.".format(
                            expected=expected,
                            colname=colname,
                            actual=actual,
                        ),
                    )

        frame.to_sql(
            tablename,
            self.conn,
            if_exists='append',
            chunksize=50000,
        )

    def write_frame(self, tablename, frame):
        if tablename not in SQLITE_ADJUSTMENT_TABLENAMES:
            raise ValueError(
                "Adjustment table %s not in %s" % (
                    tablename,
                    SQLITE_ADJUSTMENT_TABLENAMES,
                )
            )
        if not (frame is None or frame.empty):
            frame = frame.copy()
            frame['effective_date'] = frame['effective_date'].values.astype(
                'datetime64[s]',
            ).astype('int64')
        return self._write(
            tablename,
            SQLITE_ADJUSTMENT_COLUMN_DTYPES,
            frame,
        )

    def write_dividend_payouts(self, frame):
        """
        Write dividend payout data to SQLite table `dividend_payouts`.
        """
        return self._write(
            'dividend_payouts',
            SQLITE_DIVIDEND_PAYOUT_COLUMN_DTYPES,
            frame,
        )

    def write_stock_dividend_payouts(self, frame):
        return self._write(
            'stock_dividend_payouts',
            SQLITE_STOCK_DIVIDEND_PAYOUT_COLUMN_DTYPES,
            frame,
        )

    def calc_dividend_ratios(self, dividends):
        """
        Calculate the ratios to apply to equities when looking back at pricing
        history so that the price is smoothed over the ex_date, when the market
        adjusts to the change in equity value due to upcoming dividend.

        Returns
        -------
        DataFrame
            A frame in the same format as splits and mergers, with keys
            - sid, the id of the equity
            - effective_date, the date in seconds on which to apply the ratio.
            - ratio, the ratio to apply to backwards looking pricing data.
        """
        if dividends is None or dividends.empty:
            return pd.DataFrame(np.array(
                [],
                dtype=[
                    ('sid', uint64_dtype),
                    ('effective_date', uint32_dtype),
                    ('ratio', float64_dtype),
                ],
            ))

        pricing_reader = self._equity_daily_bar_reader
        input_sids = dividends.sid.values
        unique_sids, sids_ix = np.unique(input_sids, return_inverse=True)
        dates = pricing_reader.sessions.values

        close, = pricing_reader.load_raw_arrays(
            ['close'],
            pd.Timestamp(dates[0], tz='UTC'),
            pd.Timestamp(dates[-1], tz='UTC'),
            unique_sids,
        )
        date_ix = np.searchsorted(dates, dividends.ex_date.values)
        mask = date_ix > 0

        date_ix = date_ix[mask]
        sids_ix = sids_ix[mask]
        input_dates = dividends.ex_date.values[mask]

        # subtract one day to get the close on the day prior to the merger
        previous_close = close[date_ix - 1, sids_ix]
        input_sids = input_sids[mask]

        amount = dividends.amount.values[mask]
        ratio = 1.0 - amount / previous_close

        non_nan_ratio_mask = ~np.isnan(ratio)
        for ix in np.flatnonzero(~non_nan_ratio_mask):
            log.warn(
                "Couldn't compute ratio for dividend"
                " sid={sid}, ex_date={ex_date:%Y-%m-%d}, amount={amount:.3f}",
                sid=input_sids[ix],
                ex_date=pd.Timestamp(input_dates[ix]),
                amount=amount[ix],
            )

        positive_ratio_mask = ratio > 0
        for ix in np.flatnonzero(~positive_ratio_mask & non_nan_ratio_mask):
            log.warn(
                "Dividend ratio <= 0 for dividend"
                " sid={sid}, ex_date={ex_date:%Y-%m-%d}, amount={amount:.3f}",
                sid=input_sids[ix],
                ex_date=pd.Timestamp(input_dates[ix]),
                amount=amount[ix],
            )

        valid_ratio_mask = non_nan_ratio_mask & positive_ratio_mask
        return pd.DataFrame({
            'sid': input_sids[valid_ratio_mask],
            'effective_date': input_dates[valid_ratio_mask],
            'ratio': ratio[valid_ratio_mask],
        })

    def _write_dividends(self, dividends):
        if dividends is None:
            dividend_payouts = None
        else:
            dividend_payouts = dividends.copy()
            dividend_payouts['ex_date'] = dividend_payouts['ex_date'].values.\
                astype('datetime64[s]').astype(int64_dtype)
            dividend_payouts['record_date'] = \
                dividend_payouts['record_date'].values.\
                astype('datetime64[s]').astype(int64_dtype)
            dividend_payouts['declared_date'] = \
                dividend_payouts['declared_date'].values.\
                astype('datetime64[s]').astype(int64_dtype)
            dividend_payouts['pay_date'] = \
                dividend_payouts['pay_date'].values.astype('datetime64[s]').\
                astype(int64_dtype)

        self.write_dividend_payouts(dividend_payouts)

    def _write_stock_dividends(self, stock_dividends):
        if stock_dividends is None:
            stock_dividend_payouts = None
        else:
            stock_dividend_payouts = stock_dividends.copy()
            stock_dividend_payouts['ex_date'] = \
                stock_dividend_payouts['ex_date'].values.\
                astype('datetime64[s]').astype(int64_dtype)
            stock_dividend_payouts['record_date'] = \
                stock_dividend_payouts['record_date'].values.\
                astype('datetime64[s]').astype(int64_dtype)
            stock_dividend_payouts['declared_date'] = \
                stock_dividend_payouts['declared_date'].\
                values.astype('datetime64[s]').astype(int64_dtype)
            stock_dividend_payouts['pay_date'] = \
                stock_dividend_payouts['pay_date'].\
                values.astype('datetime64[s]').astype(int64_dtype)
        self.write_stock_dividend_payouts(stock_dividend_payouts)

    def write_dividend_data(self, dividends, stock_dividends=None):
        """
        Write both dividend payouts and the derived price adjustment ratios.
        """

        # First write the dividend payouts.
        self._write_dividends(dividends)
        self._write_stock_dividends(stock_dividends)

        # Second from the dividend payouts, calculate ratios.
        dividend_ratios = self.calc_dividend_ratios(dividends)
        self.write_frame('dividends', dividend_ratios)

    def write(self,
              splits=None,
              mergers=None,
              dividends=None,
              stock_dividends=None):
        """
        Writes data to a SQLite file to be read by SQLiteAdjustmentReader.

        Parameters
        ----------
        splits : pandas.DataFrame, optional
            Dataframe containing split data. The format of this dataframe is:
              effective_date : int
                  The date, represented as seconds since Unix epoch, on which
                  the adjustment should be applied.
              ratio : float
                  A value to apply to all data earlier than the effective date.
                  For open, high, low, and close those values are multiplied by
                  the ratio. Volume is divided by this value.
              sid : int
                  The asset id associated with this adjustment.
        mergers : pandas.DataFrame, optional
            DataFrame containing merger data. The format of this dataframe is:
              effective_date : int
                  The date, represented as seconds since Unix epoch, on which
                  the adjustment should be applied.
              ratio : float
                  A value to apply to all data earlier than the effective date.
                  For open, high, low, and close those values are multiplied by
                  the ratio. Volume is unaffected.
              sid : int
                  The asset id associated with this adjustment.
        dividends : pandas.DataFrame, optional
            DataFrame containing dividend data. The format of the dataframe is:
              sid : int
                  The asset id associated with this adjustment.
              ex_date : datetime64
                  The date on which an equity must be held to be eligible to
                  receive payment.
              declared_date : datetime64
                  The date on which the dividend is announced to the public.
              pay_date : datetime64
                  The date on which the dividend is distributed.
              record_date : datetime64
                  The date on which the stock ownership is checked to determine
                  distribution of dividends.
              amount : float
                  The cash amount paid for each share.

            Dividend ratios are calculated as:
            ``1.0 - (dividend_value / "close on day prior to ex_date")``
        stock_dividends : pandas.DataFrame, optional
            DataFrame containing stock dividend data. The format of the
            dataframe is:
              sid : int
                  The asset id associated with this adjustment.
              ex_date : datetime64
                  The date on which an equity must be held to be eligible to
                  receive payment.
              declared_date : datetime64
                  The date on which the dividend is announced to the public.
              pay_date : datetime64
                  The date on which the dividend is distributed.
              record_date : datetime64
                  The date on which the stock ownership is checked to determine
                  distribution of dividends.
              payment_sid : int
                  The asset id of the shares that should be paid instead of
                  cash.
              ratio : float
                  The ratio of currently held shares in the held sid that
                  should be paid with new shares of the payment_sid.

        See Also
        --------
        zipline.data.adjustments.SQLiteAdjustmentReader
        """
        self.write_frame('splits', splits)
        self.write_frame('mergers', mergers)
        self.write_dividend_data(dividends, stock_dividends)
        # Use IF NOT EXISTS here to allow multiple writes if desired.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS splits_sids "
            "ON splits(sid)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS splits_effective_date "
            "ON splits(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS mergers_sids "
            "ON mergers(sid)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS mergers_effective_date "
            "ON mergers(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS dividends_sid "
            "ON dividends(sid)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS dividends_effective_date "
            "ON dividends(effective_date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS dividend_payouts_sid "
            "ON dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS dividends_payouts_ex_date "
            "ON dividend_payouts(ex_date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS stock_dividend_payouts_sid "
            "ON stock_dividend_payouts(sid)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS stock_dividends_payouts_ex_date "
            "ON stock_dividend_payouts(ex_date)"
        )



if __name__ == '__main__':

    metadata = MetaData()

    engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/test',
                           isolation_level="READ UNCOMMITTED")
    # db = 'db'
    # with engine.connect() as conn:
    #     conn.execute('create database %s'%db)
    #
    # metadata.create_all(bind = engine)
    #
    # # engine.execution_options()
    print(metadata.clear())
    tbls = engine.table_names()
    print(tbls)
    # conn = engine.connect()
    # res = conn.execution_options(isolation_level="READ COMMITTED")
    # """
    # READ COMMITTED
    # READ UNCOMMITTED
    # REPEATABLE READ
    # SERIALIZABLE
    # AUTOCOMMIT
    # """
    # print(res.get_execution_options())
    # # engine.execution_options(isolation_level="READ COMMITTED")
    # # print(engine.get_execution_options())
    # #代理
    # from sqlalchemy import inspect
    # insp = inspect(engine)
    # print('insp',insp)
    # print(insp.get_table_names())
    # print(insp.get_columns('asharePrice'))
    # # get_pk_constraint get_primary_keys get_foreign_keys get_indexes
    # print(insp.get_schema_names())
