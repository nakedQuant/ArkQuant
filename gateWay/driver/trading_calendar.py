from sqlalchemy import select,and_,cast ,Numeric,desc,Integer


class Calendar(object):
    """
        1 交易日
        2 节假日
        3 lifeSpan
    """

    def __init__(self,conn):
        self.conn = conn

    def load_calendar(self,sdate,edate):
        """获取交易日"""
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt]).where(table.c.trade_dt.between(sdate,edate))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt

    def is_calendar(self,dt):
        """判断是否为交易日"""
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt])
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        flag = dt in trade_dt
        return flag

    def load_calendar_offset(self,date,sid):
        table = self.tables['ashareCalendar']
        if sid > 0 :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt > date)
            ins = ins.order_by(table.c.trade_dt)
        else :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt < date)
            ins = ins.order_by(desc(table.c.trade_dt))
        ins = ins.limit(abs(sid))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt

    def retrieve_symbol_lifeSpan(self,asset):
        """
            判断是否退市或者停止交易
        """
        table = self.tables['symbol_lifeSpan']
        span = sa.select([table.c.delist_date]).\
            where(table.c.code == asset).execute().scalar()
        return span

    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        pass

    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        pass

    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        pass

    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        """
        pass

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest minute on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded minute.
        dt : pd.Timestamp
            The minute at which to start searching for the last traded minute.

        Returns
        -------
        last_traded : pd.Timestamp
            The dt of the last trade for the given asset, using the input
            dt as a vantage point.
        """
        pass

    def _dt_window_size(self, start_dt, end_dt):
        pass
