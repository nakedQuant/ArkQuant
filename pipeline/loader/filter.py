"""
filter.py
"""
from itertools import chain
from toolz import flip
import datetime
from numpy import (
    float64,
    nan,
    nanpercentile,
    uint8,
    isnan,
    dtype,
    datetime64
)

uint8_dtype = dtype('uint8')
bool_dtype = dtype('bool')

uint32_dtype = dtype('uint32')
uint64_dtype = dtype('uint64')
int64_dtype = dtype('int64')

float32_dtype = dtype('float32')
float64_dtype = dtype('float64')

complex128_dtype = dtype('complex128')

datetime64D_dtype = dtype('datetime64[D]')
datetime64ns_dtype = dtype('datetime64[ns]')

object_dtype = dtype('O')
# We use object arrays for strings.
categorical_dtype = object_dtype

make_datetime64ns = flip(datetime64, 'ns')
make_datetime64D = flip(datetime64, 'D')

CLASSIFIER_DTYPES = frozenset({object_dtype, int64_dtype})
FACTOR_DTYPES = frozenset({datetime64ns_dtype, float64_dtype, int64_dtype})

def make_kind_check(python_types, numpy_kind):
    """
    Make a function that checks whether a scalar or array is of a given kind
    (e.g. float, int, datetime, timedelta).
    """
    def check(value):
        if hasattr(value, 'dtype'):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)
    return check


is_float = make_kind_check(float, 'f')
is_int = make_kind_check(int, 'i')
is_datetime = make_kind_check(datetime, 'M')
is_object = make_kind_check(object, 'O')

def isnat(obj):
    """
    Check if a value is np.NaT.
    """
    if obj.dtype.kind not in ('m', 'M'):
        raise ValueError("%s is not a numpy datetime or timedelta")
    return obj.view(int64_dtype) == iNaT


def is_missing(data, missing_value):
    """
    Generic is_missing function that handles NaN and NaT.
    """
    if is_float(data) and isnan(missing_value):
        return isnan(data)
    elif is_datetime(data) and isnat(missing_value):
        return isnat(data)
    return (data == missing_value)


def concat_tuples(*tuples):
    """
    Concatenate a sequence of tuples into one tuple.
    """
    return tuple(chain(*tuples))

bool_dtype = dtype('bool')
FILTER_DTYPES = frozenset({bool_dtype})


class Filter(object):
    """
    Pipeline expression computing a boolean output.

    Filters are most commonly useful for describing sets of assets to include
    or exclude for some particular purpose. Many Pipeline API functions accept
    a ``mask`` argument, which can be supplied a Filter indicating that only
    values passing the Filter should be considered when performing the
    requested computation. For example, :meth:`zipline.pipeline.Factor.top`
    accepts a mask indicating that ranks should be computed only on assets that
    passed the specified Filter.

    The most common way to construct a Filter is via one of the comparison
    operators (``<``, ``<=``, ``!=``, ``eq``, ``>``, ``>=``) of


    Filters can be combined via the ``&`` (and) and ``|`` (or) operators.

    ``&``-ing together two filters produces a new Filter that produces True if
    **both** of the inputs produced True.

    ``|``-ing together two filters produces a new Filter that produces True if
    **either** of its inputs produced True.

    The ``~`` operator can be used to invert a Filter, swapping all True values
    with Falses and vice-versa.

    Filters may be set as the ``screen`` attribute of a Pipeline, indicating
    asset/date pairs for which the filter produces False should be excluded
    from the Pipeline's output.  This is useful both for reducing noise in the
    output of a Pipeline and for reducing memory consumption of Pipeline
    results.
    """
    # Filters are window-safe by default, since a yes/no decision means the
    # same thing from all temporal perspectives.
    window_safe = True

    # Used by RestrictedDTypeMixin
    ALLOWED_DTYPES = FILTER_DTYPES
    dtype = bool_dtype

    def _validate(self):
        # Run superclass validation first so that we handle `dtype not passed`
        # before this.
        retval = super(Filter, self)._validate()
        if self.dtype != bool_dtype:
            raise ValueError('filter values --- bool values')
        return retval


class NullFilter(Filter):
    """
    A Filter indicating whether input values are missing from an input.

    Parameters
    ----------
    factor : zipline.pipeline.Term
        The factor to compare against its missing_value.
    """
    window_length = 0

    def __new__(cls, missing_value):
        return super(NullFilter, cls).__new__(
            cls,
            missing_value,
        )

    def _compute(self, arrays, dates, assets, mask):
        return is_missing(arrays, self.missing_value)


class NotNullFilter(Filter):
    """
    A Filter indicating whether input values are **not** missing from an input.

    Parameters
    ----------
    factor : zipline.pipeline.Term
        The factor to compare against its missing_value.
    """
    window_length = 0

    def __new__(cls, missing_value):
        return super(NotNullFilter, cls).__new__(
            cls,
            missing_value,
        )

    def _compute(self, arrays, dates, assets, mask):
        return ~is_missing(arrays, self.missing_value)


class PercentileFilter(Filter):
    """
    A Filter representing assets falling between percentile bounds of a Factor.

    Parameters
    ----------
    factor : zipline.pipeline.factor.Factor
        The factor over which to compute percentile bounds.
    min_percentile : float [0.0, 1.0]
        The minimum percentile rank of an asset that will pass the filter.
    max_percentile : float [0.0, 1.0]
        The maxiumum percentile rank of an asset that will pass the filter.
    """

    def __new__(cls, min_percentile, max_percentile, mask):
        return super(PercentileFilter, cls).__new__(
            cls,
            mask=mask,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )

    def _init(self, min_percentile, max_percentile, *args, **kwargs):
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile
        return super(PercentileFilter, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, min_percentile, max_percentile, *args, **kwargs):
        return (
            super(PercentileFilter, cls)._static_identity(*args, **kwargs),
            min_percentile,
            max_percentile,
        )

    def _validate(self):
        """
        Ensure that our percentile bounds are well-formed.
        """
        if not 0.0 <= self._min_percentile < self._max_percentile <= 100.0:
            raise BadPercentileBounds(
                min_percentile=self._min_percentile,
                max_percentile=self._max_percentile,
                upper_bound=100.0
            )
        return super(PercentileFilter, self)._validate()

    def _compute(self, arrays, dates, assets, mask):
        """
        For each row in the input, compute a mask of all values falling between
        the given percentiles.
        """
        data = arrays.copy().astype(float64)
        data[~mask] = nan

        lower_bounds = nanpercentile(
            data,
            self._min_percentile,
            axis=1,
            keepdims=True,
        )
        upper_bounds = nanpercentile(
            data,
            self._max_percentile,
            axis=1,
            keepdims=True,
        )
        return (lower_bounds <= data) & (data <= upper_bounds)

class ArrayPredicate(Filter):
    """
    A filter applying a function from (ndarray, *args) -> ndarray[bool].

    Parameters
    ----------
    term : zipline.pipeline.Term
        Term producing the array over which the predicate will be computed.
    op : function(ndarray, *args) -> ndarray[bool]
        Function to apply to the result of `term`.
    opargs : tuple[hashable]
        Additional argument to apply to ``op``.
    """
    params = ('op', 'opargs')
    window_length = 0

    # @expect_types(term=Term, opargs=tuple)
    def __new__(cls, op, opargs):
        hash(opargs)  # fail fast if opargs isn't hashable.
        return super(ArrayPredicate, cls).__new__(
            ArrayPredicate,
            op=op,
            opargs=opargs,
        )

    def _compute(self, arrays, dates, assets, mask):
        params = self.params
        return params['op'](arrays, *params['opargs']) & mask
