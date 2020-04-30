"""
    1. adjustments 与 price 分离
    2. 构建price ajustments 接口
    3. 不同数据源接口 : 数据库 h5 csv json
    4. 读取数据的方式 : bcolz blaze
    5. 构建管道的数据接口

    资产类别: 股票 ETF 基准 ， 其中可转债 | H股作为股票属性

    碰到问题是否需要将spider数据爬取与数据入库分离
    spider作为数据下载端入口，现在问题数据入口端

    data_portal --- asset.roller_finder.py
                --- reader.py
                --- dispatcher_bar_reader.py
                --- history_loader.py
                --- resample.py

    roller_finder.py :
    from zipline.assets import (
    Asset,
    AssetConvertible,
    Equity,
    Future,
    PricingDataAssociable,

    AssertFinder

    reader.py --- data_frequency
                  --- load_raw_arrays (columns, start_date, end_date, assets)
                  --- first_trading_day
                  --- last_avaiable_dt
                  --- trading_calendar
                  --- get_value (sid, dt, field)
                  --- get_last_trade_dt (asset, dt)

)



"""

# Default number of decimal places used for rounding asset prices.
DEFAULT_ASSET_PRICE_DECIMALS = 3


class HistoryCompatibleUSEquityAdjustmentReader(object):

    def __init__(self, adjustment_reader):
        self._adjustments_reader = adjustment_reader

    def load_pricing_adjustments(self, columns, dts, assets):
        """
        Returns
        -------
        adjustments : list[dict[int -> Adjustment]]
            A list, where each element corresponds to the `columns`, of
            mappings from index to adjustment objects to apply at that index.
        """
        out = [None] * len(columns)
        for i, column in enumerate(columns):
            adjs = {}
            for asset in assets:
                adjs.update(self._get_adjustments_in_range(
                    asset, dts, column))
            out[i] = adjs
        return out

    def _get_adjustments_in_range(self, asset, dts, field):
        """
        Get the Float64Multiply objects to pass to an AdjustedArrayWindow.

        For the use of AdjustedArrayWindow in the loader, which looks back
        from current simulation time back to a window of data the dictionary is
        structured with:
        - the key into the dictionary for adjustments is the location of the
        day from which the window is being viewed.
        - the start of all multiply objects is always 0 (in each window all
          adjustments are overlapping)
        - the end of the multiply object is the location before the calendar
          location of the adjustment action, making all days before the event
          adjusted.

        Parameters
        ----------
        asset : Asset
            The assets for which to get adjustments.
        dts : iterable of datetime64-like
            The dts for which adjustment data is needed.
        field : str
            OHLCV field for which to get the adjustments.

        Returns
        -------
        out : dict[loc -> Float64Multiply]
            The adjustments as a dict of loc -> Float64Multiply
        """
        sid = int(asset)
        start = normalize_date(dts[0])
        end = normalize_date(dts[-1])
        adjs = {}
        if field != 'volume':
            mergers = self._adjustments_reader.get_adjustments_for_sid(
                'mergers', sid)
            for m in mergers:
                dt = m[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    adj_loc = end_loc
                    mult = Float64Multiply(0,
                                           end_loc - 1,
                                           0,
                                           0,
                                           m[1])
                    try:
                        adjs[adj_loc].append(mult)
                    except KeyError:
                        adjs[adj_loc] = [mult]
            divs = self._adjustments_reader.get_adjustments_for_sid(
                'dividends', sid)
            for d in divs:
                dt = d[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    adj_loc = end_loc
                    mult = Float64Multiply(0,
                                           end_loc - 1,
                                           0,
                                           0,
                                           d[1])
                    try:
                        adjs[adj_loc].append(mult)
                    except KeyError:
                        adjs[adj_loc] = [mult]
        splits = self._adjustments_reader.get_adjustments_for_sid(
            'splits', sid)
        for s in splits:
            dt = s[0]
            if start < dt <= end:
                if field == 'volume':
                    ratio = 1.0 / s[1]
                else:
                    ratio = s[1]
                end_loc = dts.searchsorted(dt)
                adj_loc = end_loc
                mult = Float64Multiply(0,
                                       end_loc - 1,
                                       0,
                                       0,
                                       ratio)
                try:
                    adjs[adj_loc].append(mult)
                except KeyError:
                    adjs[adj_loc] = [mult]
        return adjs

class SlidingWindow(object):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """

    def __init__(self, window, size, cal_start, offset):
        self.window = window
        self.cal_start = cal_start
        self.current = next(window)
        self.offset = offset
        self.most_recent_ix = self.cal_start + size

    def get(self, end_ix):
        """
        Returns
        -------
        out : A np.ndarray of the equity pricing up to end_ix after adjustments
              and rounding have been applied.
        """
        if self.most_recent_ix == end_ix:
            return self.current

        target = end_ix - self.cal_start - self.offset + 1
        self.current = self.window.seek(target)

        self.most_recent_ix = end_ix
        return self.current


class HistoryLoader(with_metaclass(ABCMeta)):
    """
    loader for sliding history windows, with support for adjustments.

    Parameters
    ----------
    trading_calendar: TradingCalendar
        Contains the grouping logic needed to assign minutes to periods.
    reader : DailyBarReader, MinuteBarReader
        Reader for pricing bars.
    adjustment_reader : SQLiteAdjustmentReader
        Reader for adjustment data.
    """
    FIELDS = ('open', 'high', 'low', 'close', 'volume', 'sid')

    def __init__(self, trading_calendar, reader, equity_adjustment_reader,
                 asset_finder,
                 roll_finders=None,
                 sid_cache_size=1000,
                 prefetch_length=0):
        self.trading_calendar = trading_calendar
        self._asset_finder = asset_finder
        self._reader = reader
        self._adjustment_readers = {}
        if equity_adjustment_reader is not None:
            self._adjustment_readers[Equity] = \
                HistoryCompatibleUSEquityAdjustmentReader(
                    equity_adjustment_reader)
        if roll_finders:
            self._adjustment_readers[ContinuousFuture] =\
                ContinuousFutureAdjustmentReader(trading_calendar,
                                                 asset_finder,
                                                 reader,
                                                 roll_finders,
                                                 self._frequency)
        self._window_blocks = {
            field: ExpiringCache(LRU(sid_cache_size))
            for field in self.FIELDS
        }
        self._prefetch_length = prefetch_length

    @abstractproperty
    def _frequency(self):
        pass

    @abstractproperty
    def _calendar(self):
        pass

    @abstractmethod
    def _array(self, start, end, assets, field):
        pass

    def _decimal_places_for_asset(self, asset, reference_date):
        if isinstance(asset, Future) and asset.tick_size:
            return number_of_decimal_places(asset.tick_size)
        elif isinstance(asset, ContinuousFuture):
            # Tick size should be the same for all contracts of a continuous
            # future, so arbitrarily get the contract with next upcoming auto
            # close date.
            oc = self._asset_finder.get_ordered_contracts(asset.root_symbol)
            contract_sid = oc.contract_before_auto_close(reference_date.value)
            if contract_sid is not None:
                contract = self._asset_finder.retrieve_asset(contract_sid)
                if contract.tick_size:
                    return number_of_decimal_places(contract.tick_size)
        return DEFAULT_ASSET_PRICE_DECIMALS

    def _ensure_sliding_windows(self, assets, dts, field,
                                is_perspective_after):
        """
        Ensure that there is a Float64Multiply window for each asset that can
        provide data for the given parameters.
        If the corresponding window for the (assets, len(dts), field) does not
        exist, then create a new one.
        If a corresponding window does exist for (assets, len(dts), field), but
        can not provide data for the current dts range, then create a new
        one and replace the expired window.

        Parameters
        ----------
        assets : iterable of assets
            The assets in the window
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str
            The OHLCV field for which to retrieve data.
        is_perspective_after : bool
            see: `PricingHistoryLoader.history`

        Returns
        -------
        out : list of Float64Window with sufficient data so that each asset's
        window can provide `get` for the index corresponding with the last
        value in `dts`
        """
        end = dts[-1]
        size = len(dts)
        asset_windows = {}
        needed_assets = []
        cal = self._calendar

        assets = self._asset_finder.retrieve_all(assets)
        end_ix = find_in_sorted_index(cal, end)

        for asset in assets:
            try:
                window = self._window_blocks[field].get(
                    (asset, size, is_perspective_after), end)
            except KeyError:
                needed_assets.append(asset)
            else:
                if end_ix < window.most_recent_ix:
                    # Window needs reset. Requested end index occurs before the
                    # end index from the previous history call for this window.
                    # Grab new window instead of rewinding adjustments.
                    needed_assets.append(asset)
                else:
                    asset_windows[asset] = window

        if needed_assets:
            offset = 0
            start_ix = find_in_sorted_index(cal, dts[0])

            prefetch_end_ix = min(end_ix + self._prefetch_length, len(cal) - 1)
            prefetch_end = cal[prefetch_end_ix]
            prefetch_dts = cal[start_ix:prefetch_end_ix + 1]
            if is_perspective_after:
                adj_end_ix = min(prefetch_end_ix + 1, len(cal) - 1)
                adj_dts = cal[start_ix:adj_end_ix + 1]
            else:
                adj_dts = prefetch_dts
            prefetch_len = len(prefetch_dts)
            array = self._array(prefetch_dts, needed_assets, field)

            if field == 'sid':
                window_type = Int64Window
            else:
                window_type = Float64Window

            view_kwargs = {}
            if field == 'volume':
                array = array.astype(float64_dtype)

            for i, asset in enumerate(needed_assets):
                adj_reader = None
                try:
                    adj_reader = self._adjustment_readers[type(asset)]
                except KeyError:
                    adj_reader = None
                if adj_reader is not None:
                    adjs = adj_reader.load_pricing_adjustments(
                        [field], adj_dts, [asset])[0]
                else:
                    adjs = {}
                window = window_type(
                    array[:, i].reshape(prefetch_len, 1),
                    view_kwargs,
                    adjs,
                    offset,
                    size,
                    int(is_perspective_after),
                    self._decimal_places_for_asset(asset, dts[-1]),
                )
                sliding_window = SlidingWindow(window, size, start_ix, offset)
                asset_windows[asset] = sliding_window
                self._window_blocks[field].set(
                    (asset, size, is_perspective_after),
                    sliding_window,
                    prefetch_end)

        return [asset_windows[asset] for asset in assets]

    def history(self, assets, dts, field, is_perspective_after):
        """
        A window of pricing data with adjustments applied assuming that the
        end of the window is the day before the current simulation time.

        Parameters
        ----------
        assets : iterable of assets
            The assets in the window.
        dts : iterable of datetime64-like
            The datetimes for which to fetch data.
            Makes an assumption that all dts are present and contiguous,
            in the calendar.
        field : str
            The OHLCV field for which to retrieve data.
        is_perspective_after : bool
            True, if the window is being viewed immediately after the last dt
            in the sliding window.
            False, if the window is viewed on the last dt.

            This flag is used for handling the case where the last dt in the
            requested window immediately precedes a corporate action, e.g.:

            - is_perspective_after is True

            When the viewpoint is after the last dt in the window, as when a
            daily history window is accessed from a simulation that uses a
            minute data frequency, the history call to this loader will not
            include the current simulation dt. At that point in time, the raw
            data for the last day in the window will require adjustment, so the
            most recent adjustment with respect to the simulation time is
            applied to the last dt in the requested window.

            An example equity which has a 0.5 split ratio dated for 05-27,
            with the dts for a history call of 5 bars with a '1d' frequency at
            05-27 9:31. Simulation frequency is 'minute'.

            (In this case this function is called with 4 daily dts, and the
             calling function is responsible for stitching back on the
             'current' dt)

            |       |       |       |       | last dt | <-- viewer is here |
            |       | 05-23 | 05-24 | 05-25 | 05-26   | 05-27 9:31         |
            | raw   | 10.10 | 10.20 | 10.30 | 10.40   |                    |
            | adj   |  5.05 |  5.10 |  5.15 |  5.25   |                    |

            The adjustment is applied to the last dt, 05-26, and all previous
            dts.

            - is_perspective_after is False, daily

            When the viewpoint is the same point in time as the last dt in the
            window, as when a daily history window is accessed from a
            simulation that uses a daily data frequency, the history call will
            include the current dt. At that point in time, the raw data for the
            last day in the window will be post-adjustment, so no adjustment
            is applied to the last dt.

            An example equity which has a 0.5 split ratio dated for 05-27,
            with the dts for a history call of 5 bars with a '1d' frequency at
            05-27 0:00. Simulation frequency is 'daily'.

            |       |       |       |       |       | <-- viewer is here |
            |       |       |       |       |       | last dt            |
            |       | 05-23 | 05-24 | 05-25 | 05-26 | 05-27              |
            | raw   | 10.10 | 10.20 | 10.30 | 10.40 | 5.25               |
            | adj   |  5.05 |  5.10 |  5.15 |  5.20 | 5.25               |

            Adjustments are applied 05-23 through 05-26 but not to the last dt,
            05-27

        Returns
        -------
        out : np.ndarray with shape(len(days between start, end), len(assets))
        """
        block = self._ensure_sliding_windows(assets,
                                             dts,
                                             field,
                                             is_perspective_after)
        end_ix = self._calendar.searchsorted(dts[-1])

        return concatenate(
            [window.get(end_ix) for window in block],
            axis=1,
        )


class DailyHistoryLoader(HistoryLoader):

    @property
    def _frequency(self):
        return 'daily'

    @property
    def _calendar(self):
        return self._reader.sessions

    def _array(self, dts, assets, field):
        return self._reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )[0]


class MinuteHistoryLoader(HistoryLoader):

    @property
    def _frequency(self):
        return 'minute'

    @lazyval
    def _calendar(self):
        mm = self.trading_calendar.all_minutes
        start = mm.searchsorted(self._reader.first_trading_day)
        end = mm.searchsorted(self._reader.last_available_dt, side='right')
        return mm[start:end]

    def _array(self, dts, assets, field):
        return self._reader.load_raw_arrays(
            [field],
            dts[0],
            dts[-1],
            assets,
        )[0]

