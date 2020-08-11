

class MaxDrawdown(SingleInputMixin, CustomFactor):
    """
    Max Drawdown

    **Default Inputs:** None

    **Default Window Length:** None
    """
    ctx = ignore_nanwarnings()

    def compute(self, today, assets, out, data):
        drawdowns = fmax.accumulate(data, axis=0) - data
        drawdowns[isnan(drawdowns)] = NINF
        drawdown_ends = nanargmax(drawdowns, axis=0)

        # TODO: Accelerate this loop in Cython or Numba.
        for i, end in enumerate(drawdown_ends):
            peak = nanmax(data[:end + 1, i])
            out[i] = (peak - data[end, i]) / data[end, i]

def exponential_weights(length, decay_rate):
    """
    Build a weight vector for an exponentially-weighted statistic.

    The resulting ndarray is of the form::

        [decay_rate ** length, ..., decay_rate ** 2, decay_rate]

    Parameters
    ----------
    length : int
        The length of the desired weight vector.
    decay_rate : float
        The rate at which entries in the weight vector increase or decrease.

    Returns
    -------
    weights : ndarray[float64]
    """
    return full(length, decay_rate, float64_dtype) ** arange(length + 1, 1, -1)


decay_rate = (1.0 - (2.0 / (1.0 + span)))
decay_rate = exp(log(.5) / halflife)
assert 0.0 < decay_rate <= 1.0

_, inverse, counts = unique(  # Get counts, idx of unique groups
    group_labels,
    return_counts=True,
    return_inverse=True,
)

#clip --- smaller than a become a ; bigger than b become b
class RSI:
    """
    Relative Strength Index

    **Default Inputs**: [EquityPricing.close]

    **Default Window Length**: 15
    """
    window_length = 15
    inputs = (EquityPricing.close,)
    window_safe = True

    def compute(self, today, assets, out, closes):
        diffs = diff(closes, axis=0)
        ups = nanmean(clip(diffs, 0, inf), axis=0)
        downs = abs(nanmean(clip(diffs, -inf, 0), axis=0))
        return evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            local_dict={'ups': ups, 'downs': downs},
            global_dict={},
            out=out,
        )

class Aroon:
    """
    Aroon technical indicator.
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/aroon-indicator  # noqa

    **Defaults Inputs:** EquityPricing.low, EquityPricing.high

    Parameters
    ----------
    window_length : int > 0
        Length of the lookback window over which to compute the Aroon
        indicator.
    """

    inputs = (EquityPricing.low, EquityPricing.high)
    outputs = ('down', 'up')

    def compute(self, today, assets, out, lows, highs):
        wl = self.window_length
        high_date_index = nanargmax(highs, axis=0)
        low_date_index = nanargmin(lows, axis=0)
        evaluate(
            '(100 * high_date_index) / (wl - 1)',
            local_dict={
                'high_date_index': high_date_index,
                'wl': wl,
            },
            out=out.up,
        )
        evaluate(
            '(100 * low_date_index) / (wl - 1)',
            local_dict={
                'low_date_index': low_date_index,
                'wl': wl,
            },
            out=out.down,
        )

class FastStochasticOscillator(CustomFactor):
    """
    Fast Stochastic Oscillator Indicator [%K, Momentum Indicator]
    https://wiki.timetotrade.eu/Stochastic

    This stochastic is considered volatile, and varies a lot when used in
    market analysis. It is recommended to use the slow stochastic oscillator
    or a moving average of the %K [%D].

    **Default Inputs:** :data: `zipline.pipeline.data.EquityPricing.close`
                        :data: `zipline.pipeline.data.EquityPricing.low`
                        :data: `zipline.pipeline.data.EquityPricing.high`

    **Default Window Length:** 14

    Returns
    -------
    out: %K oscillator
    """
    inputs = (EquityPricing.close, EquityPricing.low, EquityPricing.high)
    window_safe = True
    window_length = 14

    def compute(self, today, assets, out, closes, lows, highs):

        highest_highs = nanmax(highs, axis=0)
        lowest_lows = nanmin(lows, axis=0)
        today_closes = closes[-1]

        evaluate(
            '((tc - ll) / (hh - ll)) * 100',
            local_dict={
                'tc': today_closes,
                'll': lowest_lows,
                'hh': highest_highs,
            },
            global_dict={},
            out=out,
        )


class MovingAverageConvergenceDivergenceSignal(CustomFactor):
    """
    Moving Average Convergence/Divergence (MACD) Signal line
    https://en.wikipedia.org/wiki/MACD

    A technical indicator originally developed by Gerald Appel in the late
    1970's. MACD shows the relationship between two moving averages and
    reveals changes in the strength, direction, momentum, and duration of a
    trend in a stock's price.

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.close`

    Parameters
    ----------
    fast_period : int > 0, optional
        The window length for the "fast" EWMA. Default is 12.
    slow_period : int > 0, > fast_period, optional
        The window length for the "slow" EWMA. Default is 26.
    signal_period : int > 0, < fast_period, optional
        The window length for the signal line. Default is 9.

    Notes
    -----
    Unlike most pipeline expressions, this factor does not accept a
    ``window_length`` parameter. ``window_length`` is inferred from
    ``slow_period`` and ``signal_period``.
    """
    inputs = (EquityPricing.close,)
    # We don't use the default form of `params` here because we want to
    # dynamically calculate `window_length` from the period lengths in our
    # __new__.
    params = ('fast_period', 'slow_period', 'signal_period')

    @expect_bounded(
        __funcname='MACDSignal',
        fast_period=(1, None),  # These must all be >= 1.
        slow_period=(1, None),
        signal_period=(1, None),
    )
    def __new__(cls,
                fast_period=12,
                slow_period=26,
                signal_period=9,
                *args,
                **kwargs):

        if slow_period <= fast_period:
            raise ValueError(
                "'slow_period' must be greater than 'fast_period', but got\n"
                "slow_period={slow}, fast_period={fast}".format(
                    slow=slow_period,
                    fast=fast_period,
                )
            )

        return super(MovingAverageConvergenceDivergenceSignal, cls).__new__(
            cls,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            window_length=slow_period + signal_period - 1,
            *args, **kwargs
        )

    def _ewma(self, data, length):
        decay_rate = 1.0 - (2.0 / (1.0 + length))
        return average(
            data,
            axis=1,
            weights=exponential_weights(length, decay_rate)
        )

    def compute(self, today, assets, out, close, fast_period, slow_period,
                signal_period):
        slow_EWMA = self._ewma(
            rolling_window(close, slow_period),
            slow_period
        )
        fast_EWMA = self._ewma(
            rolling_window(close, fast_period)[-signal_period:],
            fast_period
        )
        macd = fast_EWMA - slow_EWMA
        out[:] = self._ewma(macd.T, signal_period)


# Convenience aliases.
MACDSignal = MovingAverageConvergenceDivergenceSignal
