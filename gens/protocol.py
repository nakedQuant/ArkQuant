class BarData:
    """
    Provides methods for accessing minutely and daily price/volume data from
    Algorithm API functions.

    Also provides utility methods to determine if an asset is alive, and if it
    has recent trade data.

    An instance of this object is passed as ``data`` to
    :func:`~zipline.api.handle_data` and
    :func:`~zipline.api.before_trading_start`.

    Parameters
    ----------
    data_portal : DataPortal
        Provider for bar pricing data.
    simulation_dt_func : callable
        Function which returns the current simulation time.
        This is usually bound to a method of TradingSimulation.
    data_frequency : {'minute', 'daily'}
        The frequency of the bar data; i.e. whether the data is
        daily or minute bars
    restrictions : zipline.finance.asset_restrictions.Restrictions
        Object that combines and returns restricted list information from
        multiple sources
    universe_func : callable, optional
        Function which returns the current 'universe'.  This is for
        backwards compatibility with older API concepts.
    """

    def __init__(self, data_portal, data_frequency,
                 trading_calendar, restrictions, universe_func=None):
        self.data_portal = data_portal
        self.data_frequency = data_frequency
        self._universe_func = universe_func
        self._trading_calendar = trading_calendar
        self._is_restricted = restrictions.is_restricted

    def get_current_ticker(self,assets,fields):
        """
        Returns the "current" value of the given fields for the given assets
        at the current simulation time.
        :param assets: asset_type
        :param fields: OHLCTV
        :return: dict asset -> ticker
        intended to return current ticker
        """
        cur = {}
        for asset in assets:
            ticker = self.data_portal.get_current(asset)
            cur[asset] = ticker.loc[:,fields]
        return cur

    def history(self, assets, end_dt,bar_count, fields,frequency):
        """
        Returns a trailing window of length ``bar_count`` containing data for
        the given assets, fields, and frequency.

        Returned data is adjusted for splits, dividends, and mergers as of the
        current simulation time.

        The semantics for missing data are identical to the ones described in
        the notes for :meth:`current`.

        Parameters
        ----------
        assets: zipline.assets.Asset or iterable of zipline.assets.Asset
            The asset(s) for which data is requested.
        fields: string or iterable of string.
            Requested data field(s). Valid field names are: "price",
            "last_traded", "open", "high", "low", "close", and "volume".
        bar_count: int
            Number of data observations requested.
        frequency: str
            String indicating whether to load daily or minutely data
            observations. Pass '1m' for minutely data, '1d' for daily data.

        Returns
        -------
        history : pd.Series or pd.DataFrame or pd.Panel
            See notes below.

        Notes
        ------
        returned panel has:
        items: fields
        major axis: dt
        minor axis: assets
        return pd.Panel(df_dict)
        """
        sliding_window = self.data_portal.get_history_window(assets,
                                                             end_dt,
                                                             bar_count,
                                                             fields,
                                                             frequency)
        return sliding_window

    def window_data(self,assets,end_dt,bar_count,fields,frequency):
        window_array = self.data_portal.get_window_data(assets,
                                                        end_dt,
                                                        bar_count,
                                                        fields,
                                                        frequency)
        return window_array

    def _get_equity_price_view(self, asset):
        """
        Returns a DataPortalSidView for the given asset.  Used to support the
        data[sid(N)] public API.  Not needed if DataPortal is used standalone.

        Parameters
        ----------
        asset : Asset
            Asset that is being queried.

        Returns
        -------
        SidView : Accessor into the given asset's data.
        """
        try:
            self._warn_deprecated("`data[sid(N)]` is deprecated. Use "
                            "`data.current`.")
            view = self._views[asset]
        except KeyError:
            try:
                asset = self.data_portal.asset_finder.retrieve_asset(asset)
            except ValueError:
                # assume fetcher
                pass
            view = self._views[asset] = self._create_sid_view(asset)

        return view

    def _create_sid_view(self, asset):
        return SidView(
            asset,
            self.data_portal,
            self.simulation_dt_func,
            self.data_frequency
        )



class SidView:

    """
    This class exists to temporarily support the deprecated data[sid(N)] API.
    """
    def __init__(self, asset, data_portal, simulation_dt_func, data_frequency):
        """
        Parameters
        ---------
        asset : Asset
            The asset for which the instance retrieves data.

        data_portal : DataPortal
            Provider for bar pricing data.

        simulation_dt_func: function
            Function which returns the current simulation time.
            This is usually bound to a method of TradingSimulation.

        data_frequency: string
            The frequency of the bar data; i.e. whether the data is
            'daily' or 'minute' bars
        """
        self.asset = asset
        self.data_portal = data_portal
        self.simulation_dt_func = simulation_dt_func
        self.data_frequency = data_frequency

    def __getattr__(self, column):
        # backwards compatibility code for Q1 API
        if column == "close_price":
            column = "close"
        elif column == "open_price":
            column = "open"
        elif column == "dt":
            return self.dt
        elif column == "datetime":
            return self.datetime
        elif column == "sid":
            return self.sid

        return self.data_portal.get_spot_value(
            self.asset,
            column,
            self.simulation_dt_func(),
            self.data_frequency
        )

    def __contains__(self, column):
        return self.data_portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)

    @property
    def sid(self):
        return self.asset

    @property
    def dt(self):
        return self.datetime

    @property
    def datetime(self):
        return self.data_portal.get_last_traded_dt(
            self.asset,
            self.simulation_dt_func(),
            self.data_frequency)

    @property
    def current_dt(self):
        return self.simulation_dt_func()

    def mavg(self, num_minutes):
        self._warn_deprecated("The `mavg` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "mavg", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def stddev(self, num_minutes):
        self._warn_deprecated("The `stddev` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "stddev", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def vwap(self, num_minutes):
        self._warn_deprecated("The `vwap` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "vwap", self.simulation_dt_func(),
            self.data_frequency, bars=num_minutes
        )

    def returns(self):
        self._warn_deprecated("The `returns` method is deprecated.")
        return self.data_portal.get_simple_transform(
            self.asset, "returns", self.simulation_dt_func(),
            self.data_frequency
        )

    def _warn_deprecated(self, msg):
        warnings.warn(
            msg,
            category=ZiplineDeprecationWarning,
            stacklevel=1
        )