class BarReader:

    def __init__(self):
        self.loader = Core()
        self.ts = TushareClient()
        self.extra = spider_engine.ExtraOrdinary()

    def load_index_kline(self, sdate, edate,fields,asset):
        """
            返回特定时间区间日基准指数K线
        """
        fields = self._verify_fields(fields,asset)
        index = self.loader.load_kline(sdate, edate, asset, 'ashareIndex')
        index_kline = pd.DataFrame(index,columns = ['trade_dt','code','open','close','high','low','volume'])
        index_kline.index = index_kline.loc[:,'trade_dt']
        index_pd = index_kline.loc[:,fields]
        return index_pd

    def load_index_component(self,index,sdate,edate):
        """基准成分股以及对应权重 e.g. index_code='399300.SZ' """
        component = self.ts.to_ts_index_component(index,sdate,edate)
        return component


    def load_periphera_index(self, sdate, edate,fields,index, exchange):
        """us.DJI 道琼斯 us.IXIC 纳斯达克 us.INX  标普500 hkHSI 香港恒生指数 hkHSCEI 香港国企指数 hkHSCCI 香港红筹指数"""
        raw = self.extra.download_periphera_index(sdate, edate,index, exchange)
        raw.index = raw['trade_dt']
        index_price = raw.loc[:,fields]
        return index_price


def load_market_data(trading_day=None, trading_days=None, bm_symbol='SPY',
                     environ=None):
    """
    Load benchmark returns and treasury yield curves for the given calendar and
    benchmark symbol.

    Benchmarks are downloaded as a Series from IEX Trading.  Treasury curves
    are US Treasury Bond rates and are downloaded from 'www.federalreserve.gov'
    by default.  For Canadian exchanges, a loader for Canadian bonds from the
    Bank of Canada is also available.

    Results downloaded from the internet are cached in
    ~/.zipline/data. Subsequent loads will attempt to read from the cached
    files before falling back to redownload.

    Parameters
    ----------
    trading_day : pandas.CustomBusinessDay, optional
        A trading_day used to determine the latest day for which we
        expect to have data.  Defaults to an NYSE trading day.
    trading_days : pd.DatetimeIndex, optional
        A calendar of trading days.  Also used for determining what cached
        dates we should expect to have cached. Defaults to the NYSE calendar.
    bm_symbol : str, optional
        Symbol for the benchmark index to load. Defaults to 'SPY', the ticker
        for the S&P 500, provided by IEX Trading.

    Returns
    -------
    (benchmark_returns, treasury_curves) : (pd.Series, pd.DataFrame)

    Notes
    -----

    Both return values are DatetimeIndexed with values dated to midnight in UTC
    of each stored date.  The columns of `treasury_curves` are:

    '1month', '3month', '6month',
    '1year','2year','3year','5year','7year','10year','20year','30year'
    """
    if trading_day is None:
        trading_day = get_calendar('XNYS').day
    if trading_days is None:
        trading_days = get_calendar('XNYS').all_sessions

    first_date = trading_days[0]
    now = pd.Timestamp.utcnow()

    # we will fill missing benchmark data through latest trading date
    last_date = trading_days[trading_days.get_loc(now, method='ffill')]

    br = ensure_benchmark_data(
        bm_symbol,
        first_date,
        last_date,
        now,
        # We need the trading_day to figure out the close prior to the first
        # date so that we can compute returns for the first date.
        trading_day,
        environ,
    )
    tc = ensure_treasury_data(
        bm_symbol,
        first_date,
        last_date,
        now,
        environ,
    )

    # combine dt indices and reindex using ffill then bfill
    all_dt = br.index.union(tc.index)
    br = br.reindex(all_dt, method='ffill').fillna(method='bfill')
    tc = tc.reindex(all_dt, method='ffill').fillna(method='bfill')

    benchmark_returns = br[br.index.slice_indexer(first_date, last_date)]
    treasury_curves = tc[tc.index.slice_indexer(first_date, last_date)]
    return benchmark_returns, treasury_curves


def ensure_benchmark_data(symbol, first_date, last_date, now, trading_day,
                          environ=None):
    """
    Ensure we have benchmark data for `symbol` from `first_date` to `last_date`

    Parameters
    ----------
    symbol : str
        The symbol for the benchmark to load.
    first_date : pd.Timestamp
        First required date for the cache.
    last_date : pd.Timestamp
        Last required date for the cache.
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.
    trading_day : pd.CustomBusinessDay
        A trading day delta.  Used to find the day before first_date so we can
        get the close of the day prior to first_date.

    We attempt to download data unless we already have data stored at the data
    cache for `symbol` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """
    filename = get_benchmark_filename(symbol)
    data = _load_cached_data(filename, first_date, last_date, now, 'benchmark',
                             environ)
    if data is not None:
        return data

    # If no cached data was found or it was missing any dates then download the
    # necessary data.
    logger.info(
        ('Downloading benchmark data for {symbol!r} '
            'from {first_date} to {last_date}'),
        symbol=symbol,
        first_date=first_date - trading_day,
        last_date=last_date
    )

    try:
        data = get_benchmark_returns(symbol)
        data.to_csv(get_data_filepath(filename, environ))
    except (OSError, IOError, HTTPError):
        logger.exception('Failed to cache the new benchmark returns')
        raise
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn(
            ("Still don't have expected benchmark data for {symbol!r} "
                "from {first_date} to {last_date} after redownload!"),
            symbol=symbol,
            first_date=first_date - trading_day,
            last_date=last_date
        )
    return data


def ensure_treasury_data(symbol, first_date, last_date, now, environ=None):
    """
    Ensure we have treasury data from treasury module associated with
    `symbol`.

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're loading associated treasury curves.
    first_date : pd.Timestamp
        First date required to be in the cache.
    last_date : pd.Timestamp
        Last date required to be in the cache.
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.

    We attempt to download data unless we already have data stored in the cache
    for `module_name` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """
    loader_module, filename, source = INDEX_MAPPING.get(
        symbol, INDEX_MAPPING['SPY'],
    )
    first_date = max(first_date, loader_module.earliest_possible_date())

    data = _load_cached_data(filename, first_date, last_date, now, 'treasury',
                             environ)
    if data is not None:
        return data

    # If no cached data was found or it was missing any dates then download the
    # necessary data.
    logger.info(
        ('Downloading treasury data for {symbol!r} '
            'from {first_date} to {last_date}'),
        symbol=symbol,
        first_date=first_date,
        last_date=last_date
    )

    try:
        data = loader_module.get_treasury_data(first_date, last_date)
        data.to_csv(get_data_filepath(filename, environ))
    except (OSError, IOError, HTTPError):
        logger.exception('failed to cache treasury data')
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn(
            ("Still don't have expected treasury data for {symbol!r} "
                "from {first_date} to {last_date} after redownload!"),
            symbol=symbol,
            first_date=first_date,
            last_date=last_date
        )
    return data