
def _calculate_order_value_amount(self, asset, value):
    """
    Calculates how many shares/contracts to order based on the type of
    asset being ordered.
    """
    # Make sure the asset exists, and that there is a last price for it.
    # FIXME: we should use BarData's can_trade logic here, but I haven't
    # yet found a good way to do that.
    normalized_date = normalize_date(self.datetime)

    if normalized_date < asset.start_date:
        raise CannotOrderDelistedAsset(
            msg="Cannot order {0}, as it started trading on"
                " {1}.".format(asset.symbol, asset.start_date)
        )
    elif normalized_date > asset.end_date:
        raise CannotOrderDelistedAsset(
            msg="Cannot order {0}, as it stopped trading on"
                " {1}.".format(asset.symbol, asset.end_date)
        )
    else:
        last_price = \
            self.trading_client.current_data.current(asset, "price")

        if np.isnan(last_price):
            raise CannotOrderDelistedAsset(
                msg="Cannot order {0} on {1} as there is no last "
                    "price for the security.".format(asset.symbol,
                                                     self.datetime)
            )

    if tolerant_equals(last_price, 0):
        zero_message = "Price of 0 for {psid}; can't infer value".format(
            psid=asset
        )
        if self.logger:
            self.logger.debug(zero_message)
        # Don't place any order
        return 0

    value_multiplier = asset.price_multiplier

    return value / (last_price * value_multiplier)


def _can_order_asset(self, asset):
    if not isinstance(asset, Asset):
        raise UnsupportedOrderParameters(
            msg="Passing non-Asset argument to 'order()' is not supported."
                " Use 'sid()' or 'symbol()' methods to look up an Asset."
        )

    if asset.auto_close_date:
        day = normalize_date(self.get_datetime())

        if day > min(asset.end_date, asset.auto_close_date):
            # If we are after the asset's end date or auto close date, warn
            # the user that they can't place an order for this asset, and
            # return None.
            log.warn("Cannot place order for {0}, as it has de-listed. "
                     "Any existing positions for this asset will be "
                     "liquidated on "
                     "{1}.".format(asset.symbol, asset.auto_close_date))
            return False
    return True


@api_method
@disallowed_in_before_trading_start(OrderInBeforeTradingStart())
def order(self,
          asset,
          amount,
          limit_price=None,
          stop_price=None,
          style=None):
    """Place an order for a fixed number of shares.

    Parameters
    ----------
    asset : Asset
        The asset to be ordered.
    amount : int
        The amount of shares to order. If ``amount`` is positive, this is
        the number of shares to buy or cover. If ``amount`` is negative,
        this is the number of shares to sell or short.
    limit_price : float, optional
        The limit price for the order.
    stop_price : float, optional
        The stop price for the order.
    style : ExecutionStyle, optional
        The execution style for the order.

    Returns
    -------
    order_id : str or None
        The unique identifier for this order, or None if no order was
        placed.

    Notes
    -----
    The ``limit_price`` and ``stop_price`` arguments provide shorthands for
    passing common execution styles. Passing ``limit_price=N`` is
    equivalent to ``style=LimitOrder(N)``. Similarly, passing
    ``stop_price=M`` is equivalent to ``style=StopOrder(M)``, and passing
    ``limit_price=N`` and ``stop_price=M`` is equivalent to
    ``style=StopLimitOrder(N, M)``. It is an error to pass both a ``style``
    and ``limit_price`` or ``stop_price``.

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order_value`
    :func:`zipline.api.order_percent`
    """
    if not self._can_order_asset(asset):
        return None

    amount, style = self._calculate_order(asset, amount,
                                          limit_price, stop_price, style)
    return self.blotter.order(asset, amount, style)


def _calculate_order(self, asset, amount,
                     limit_price=None, stop_price=None, style=None):
    amount = self.round_order(amount)

    # Raises a ZiplineError if invalid parameters are detected.
    self.validate_order_params(asset,
                               amount,
                               limit_price,
                               stop_price,
                               style)

    # Convert deprecated limit_price and stop_price parameters to use
    # ExecutionStyle objects.
    style = self.__convert_order_params_for_blotter(asset,
                                                    limit_price,
                                                    stop_price,
                                                    style)
    return amount, style


@staticmethod
def round_order(amount):
    """
    Convert number of shares to an integer.

    By default, truncates to the integer share count that's either within
    .0001 of amount or closer to zero.

    E.g. 3.9999 -> 4.0; 5.5 -> 5.0; -5.5 -> -5.0
    """
    return int(round_if_near_integer(amount))


def validate_order_params(self,
                          asset,
                          amount,
                          limit_price,
                          stop_price,
                          style):
    """
    Helper method for validating parameters to the order API function.

    Raises an UnsupportedOrderParameters if invalid arguments are found.
    """

    if not self.initialized:
        raise OrderDuringInitialize(
            msg="order() can only be called from within handle_data()"
        )

    if style:
        if limit_price:
            raise UnsupportedOrderParameters(
                msg="Passing both limit_price and style is not supported."
            )

        if stop_price:
            raise UnsupportedOrderParameters(
                msg="Passing both stop_price and style is not supported."
            )

    for control in self.trading_controls:
        control.validate(asset,
                         amount,
                         self.portfolio,
                         self.get_datetime(),
                         self.trading_client.current_data)


@staticmethod
def __convert_order_params_for_blotter(asset,
                                       limit_price,
                                       stop_price,
                                       style):
    """
    Helper method for converting deprecated limit_price and stop_price
    arguments into ExecutionStyle instances.

    This function assumes that either style == None or (limit_price,
    stop_price) == (None, None).
    """
    if style:
        assert (limit_price, stop_price) == (None, None)
        return style
    if limit_price and stop_price:
        return StopLimitOrder(limit_price, stop_price, asset=asset)
    if limit_price:
        return LimitOrder(limit_price, asset=asset)
    if stop_price:
        return StopOrder(stop_price, asset=asset)
    else:
        return MarketOrder()


@api_method
@disallowed_in_before_trading_start(OrderInBeforeTradingStart())
def order_value(self,
                asset,
                value,
                limit_price=None,
                stop_price=None,
                style=None):
    """
    Place an order for a fixed amount of money.

    Equivalent to ``order(asset, value / data.current(asset, 'price'))``.

    Parameters
    ----------
    asset : Asset
        The asset to be ordered.
    value : float
        Amount of value of ``asset`` to be transacted. The number of shares
        bought or sold will be equal to ``value / current_price``.
    limit_price : float, optional
        Limit price for the order.
    stop_price : float, optional
        Stop price for the order.
    style : ExecutionStyle
        The execution style for the order.

    Returns
    -------
    order_id : str
        The unique identifier for this order.

    Notes
    -----
    See :func:`zipline.api.order` for more information about
    ``limit_price``, ``stop_price``, and ``style``

    See Also
    --------
    :class:`zipline.finance.execution.ExecutionStyle`
    :func:`zipline.api.order`
    :func:`zipline.api.order_percent`
    """
    if not self._can_order_asset(asset):
        return None

    amount = self._calculate_order_value_amount(asset, value)
    return self.order(asset, amount,
                      limit_price=limit_price,
                      stop_price=stop_price,
                      style=style)


@api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_percent(self,
                      asset,
                      percent,
                      limit_price=None,
                      stop_price=None,
                      style=None):
        """Place an order in the specified asset corresponding to the given
        percent of the current portfolio value.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        percent : float
            The percentage of the portfolio value to allocate to ``asset``.
            This is specified as a decimal, for example: 0.50 means 50%.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_percent_amount(asset, percent)
        return self.order(asset, amount,
                          limit_price=limit_price,
                          stop_price=stop_price,
                          style=style)

    def _calculate_order_percent_amount(self, asset, percent):
        value = self.portfolio.portfolio_value * percent
        return self._calculate_order_value_amount(asset, value)

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target(self,
                     asset,
                     target,
                     limit_price=None,
                     stop_price=None,
                     style=None):
        """Place an order to adjust a position to a target number of shares. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target number of shares and the
        current number of shares.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : int
            The desired number of shares of ``asset``.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.


        Notes
        -----
        ``order_target`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target(sid(0), 10)
           order_target(sid(0), 10)

        This code will result in 20 shares of ``sid(0)`` because the first
        call to ``order_target`` will not have been filled when the second
        ``order_target`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target_percent`
        :func:`zipline.api.order_target_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_target_amount(asset, target)
        return self.order(asset, amount,
                          limit_price=limit_price,
                          stop_price=stop_price,
                          style=style)

    def _calculate_order_target_amount(self, asset, target):
        if asset in self.portfolio.positions:
            current_position = self.portfolio.positions[asset].amount
            target -= current_position

        return target

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_value(self,
                           asset,
                           target,
                           limit_price=None,
                           stop_price=None,
                           style=None):
        """Place an order to adjust a position to a target value. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target value and the
        current value.
        If the Asset being ordered is a Future, the 'target value' calculated
        is actually the target exposure, as Futures have no 'value'.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : float
            The desired total value of ``asset``.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        ``order_target_value`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target_value(sid(0), 10)
           order_target_value(sid(0), 10)

        This code will result in 20 dollars of ``sid(0)`` because the first
        call to ``order_target_value`` will not have been filled when the
        second ``order_target_value`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target`
        :func:`zipline.api.order_target_percent`
        """
        if not self._can_order_asset(asset):
            return None

        target_amount = self._calculate_order_value_amount(asset, target)
        amount = self._calculate_order_target_amount(asset, target_amount)
        return self.order(asset, amount,
                          limit_price=limit_price,
                          stop_price=stop_price,
                          style=style)

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_percent(self, asset, target,
                             limit_price=None, stop_price=None, style=None):
        """Place an order to adjust a position to a target percent of the
        current portfolio value. If the position doesn't already exist, this is
        equivalent to placing a new order. If the position does exist, this is
        equivalent to placing an order for the difference between the target
        percent and the current percent.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : float
            The desired percentage of the portfolio value to allocate to
            ``asset``. This is specified as a decimal, for example:
            0.50 means 50%.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        ``order_target_value`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target_percent(sid(0), 10)
           order_target_percent(sid(0), 10)

        This code will result in 20% of the portfolio being allocated to sid(0)
        because the first call to ``order_target_percent`` will not have been
        filled when the second ``order_target_percent`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target`
        :func:`zipline.api.order_target_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_target_percent_amount(asset, target)
        return self.order(asset, amount,
                          limit_price=limit_price,
                          stop_price=stop_price,
                          style=style)

    def _calculate_order_target_percent_amount(self, asset, target):
        target_amount = self._calculate_order_percent_amount(asset, target)
        return self._calculate_order_target_amount(asset, target_amount)

    @api_method
    @expect_types(share_counts=pd.Series)
    @expect_dtypes(share_counts=int64_dtype)
    def batch_market_order(self, share_counts):
        """Place a batch market order for multiple assets.

        Parameters
        ----------
        share_counts : pd.Series[Asset -> int]
            Map from asset to number of shares to order for that asset.

        Returns
        -------
        order_ids : pd.Index[str]
            Index of ids for newly-created orders.
        """
        style = MarketOrder()
        order_args = [
            (asset, amount, style)
            for (asset, amount) in iteritems(share_counts)
            if amount
        ]
        return self.blotter.batch_order(order_args)

    @error_keywords(sid='Keyword argument `sid` is no longer supported for '
                        'get_open_orders. Use `asset` instead.')
    @api_method
    def get_open_orders(self, asset=None):
        """Retrieve all of the current open orders.

        Parameters
        ----------
        asset : Asset
            If passed and not None, return only the open orders for the given
            asset instead of all open orders.

        Returns
        -------
        open_orders : dict[list[Order]] or list[Order]
            If no asset is passed this will return a dict mapping Assets
            to a list containing all the open orders for the asset.
            If an asset is passed then this will return a list of the open
            orders for this asset.
        """
        if asset is None:
            return {
                key: [order.to_api_obj() for order in orders]
                for key, orders in iteritems(self.blotter.open_orders)
                if orders
            }
        if asset in self.blotter.open_orders:
            orders = self.blotter.open_orders[asset]
            return [order.to_api_obj() for order in orders]
        return []

    @api_method
    def get_order(self, order_id):
        """Lookup an order based on the order id returned from one of the
        order functions.

        Parameters
        ----------
        order_id : str
            The unique identifier for the order.

        Returns
        -------
        order : Order
            The order object.
        """
        if order_id in self.blotter.orders:
            return self.blotter.orders[order_id].to_api_obj()

    @api_method
    def cancel_order(self, order_param):
        """Cancel an open order.

        Parameters
        ----------
        order_param : str or Order
            The order_id or order object to cancel.
        """
        order_id = order_param
        if isinstance(order_param, zipline.protocol.Order):
            order_id = order_param.id

        self.blotter.cancel(order_id)
