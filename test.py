
# # from io import StringIO
# #
# # output = StringIO()
# # output.write('First line.\n')
# # contents = output.getvalue()
# # output.close()
# # fd = StringIO()
# # fd.tell()
# # fd.seek(0)
# # fd.close()
# # fd = StringIO()
# # if isinstance(data, str):
# #     fd.write(data)
# # else:
# #     for chunk in data:
# #         fd.write(chunk)
# # self.fetch_size = fd.tell()
# # fd.seek(0)
#
# # from abc import ABCMeta, abstractmethod
# # from collections import namedtuple
# # import hashlib
# # from textwrap import dedent
# # import warnings
# #
# # import numpy
# # import pandas as pd
# # from pandas import read_csv
# # import pytz
# # import requests
#
#
# # def __iter__(self):
# #     warnings.warn(
# #         'Iterating over security_lists is deprecated. Use '
# #         '`for sid in <security_list>.current_securities(dt)` instead.',
# #         category=ZiplineDeprecationWarning,
# #         stacklevel=2
# #     )
# #     return iter(self.current_securities(self.current_date()))
# #
# # def __contains__(self, item):
# #     warnings.warn(
# #         'Evaluating inclusion in security_lists is deprecated. Use '
# #         '`sid in <security_list>.current_securities(dt)` instead.',
# #         category=ZiplineDeprecationWarning,
# #         stacklevel=2
# #     )
# #     return item in self.current_securities(self.current_date())
# #


# # import inspect ,uuid
# # from functools import wraps
# #
# # def getargspec(f):
# #     full_argspec = inspect.getfullargspec(f)
# #     return inspect.ArgSpec(
# #         args=full_argspec.args,
# #         varargs=full_argspec.varargs,
# #         keywords=full_argspec.varkw,
# #         defaults=full_argspec.defaults,
# #     )
# # #
# # # print(getargspec(_parse_url))
# # NO_DEFAULT = object()
# # #
# # #
# # args, varargs, varkw, defaults = argspec = getargspec(_parse_url)
# # print('varargs',varargs)
# # print('varkw',varkw)
# # if defaults is None:
# #     defaults = ()
# # no_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
# # print('args',args)
# # print('no_defaults',no_defaults)
# # args_defaults = list(zip(args, no_defaults + defaults))
# # if varargs:
# #     args_defaults.append((varargs, NO_DEFAULT))
# # if varkw:
# #     args_defaults.append((varkw, NO_DEFAULT))
# # print('args_defaults',args_defaults)
# # print('varargs',varargs)
# # print('varkw',varkw)
# #
# # # argset = set(args) | {varargs, varkw} - {None}
# # argset = set(args) | {varargs, varkw}
# # #
# # print('argset',argset)
#
# # __qualname__
#

# #
# # assets_with_leading_nan = np.where(isnull(df.iloc[0]))[0]
# # normed_index = df.index.normalize()
# # _code_argorder = (
# #                      ('co_argcount', 'co_kwonlyargcount') if PY3 else ('co_argcount',)
# #                  ) + (
# #                      'co_nlocals',
# #                      'co_stacksize',
# #                      'co_flags',
# #                      'co_code',
# #                      'co_consts',
# #                      'co_names',
# #                      'co_varnames',
# #                      'co_filename',
# #                      'co_name',
# #                      'co_firstlineno',
# #                      'co_lnotab',
# #                      'co_freevars',
# #                      'co_cellvars',
# #                  )
# # code = new_func.__code__
# # args = {
# #     attr: getattr(code, attr)
# #     for attr in dir(code)
# #     if attr.startswith('co_')
# # }
# # # Copy the firstlineno out of the underlying function so that exceptions
# # # get raised with the correct traceback.
# # # This also makes dynamic source inspection (like IPython `??` operator)
# # # work as intended.
# # try:
# #     # Try to get the pycode object from the underlying function.
# #     original_code = func.__code__
# # except AttributeError:
# #     try:
# #         # The underlying callable was not a function, try to grab the
# #         # `__func__.__code__` which exists on method objects.
# #         original_code = func.__func__.__code__
# #     except AttributeError:
# #         # The underlying callable does not have a `__code__`. There is
# #         # nothing for us to correct.
# #         return new_func
# #
# # args['co_firstlineno'] = original_code.co_firstlineno
# # new_func.__code__ = CodeType(*map(getitem(args), _code_argorder))
# #
# # if PY3:
# #     _qualified_name = attrgetter('__qualname__')
# # else:
# #     def _qualified_name(obj):
# #         """
# #         Return the fully-qualified name (ignoring inner classes) of a type.
# #         """
# #         # If the obj has an explicitly-set __qualname__, use it.
# #         try:
# #             return getattr(obj, '__qualname__')
# #         except AttributeError:
# #             pass
# #
# #         # If not, build our own __qualname__ as best we can.
# #         module = obj.__module__
# #         if module in ('__builtin__', '__main__', 'builtins'):
# #             return obj.__name__
# #         return '.'.join([module, obj.__name__])
# # from collections import OrderedDict
# # from datetime import datetime
# # from distutils.version import StrictVersion
# #
# # import numpy as np
# # from numpy import (
# #     array_equal,
# #     broadcast,
# #     busday_count,
# #     datetime64,
# #     diff,
# #     dtype,
# #     empty,
# #     flatnonzero,
# #     hstack,
# #     isnan,
# #     nan,
# #     vectorize,
# #     where
# # )
# # flip --- 反转参数
# from toolz import flip
#
# # def load_from_directory(list_name):
# #     """
# #     To resolve the symbol in the LEVERAGED_ETF list,
# #     the date on which the symbol was in effect is needed.
# #
# #     Furthermore, to maintain a point in time record of our own maintenance
# #     of the restricted list, we need a knowledge date. Thus, restricted lists
# #     are dictionaries of datetime->symbol lists.
# #     new symbols should be entered as a new knowledge date entry.
# #
# #     This method assumes a directory structure of:
# #     SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/add.txt
# #     SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/delete.txt
# #
# #     The return value is a dictionary with:
# #     knowledge_date -> lookup_date ->
# #        {add: [symbol list], 'delete': [symbol list]}
# #     """
# #     data = {}
# #     dir_path = os.path.join(SECURITY_LISTS_DIR, list_name)
# #     for kd_name in listdir(dir_path):
# #         kd = datetime.strptime(kd_name, DATE_FORMAT).replace(
# #             tzinfo=pytz.utc)
# #         data[kd] = {}
# #         kd_path = os.path.join(dir_path, kd_name)
# #         for ld_name in listdir(kd_path):
# #             ld = datetime.strptime(ld_name, DATE_FORMAT).replace(
# #                 tzinfo=pytz.utc)
# #             data[kd][ld] = {}
# #             ld_path = os.path.join(kd_path, ld_name)
# #             for fname in listdir(ld_path):
# #                 fpath = os.path.join(ld_path, fname)
# #                 with open(fpath) as f:
# #                     symbols = f.read().splitlines()
# #                     data[kd][ld][fname] = symbols
# #
# #     return data
# # def inspect(self):
# #     """
# #     Return a string representation of the data stored in this array.
# #     """
# #     return dedent(
# #         """\
# #         Adjusted Array ({dtype}):
# #
# #         Data:
# #         {data!r}
# #
# #         Adjustments:
# #         {adjustments}
# #         """
# #     ).format(
# #         dtype=self.dtype.name,
# #         data=self.data,
# #         adjustments=self.adjustments,
# #     )
# # def last_modified_time(path):
# #     """
# #     Get the last modified time of path as a Timestamp.
# #     """
# #     return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')
# #
# # def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
# #     data = pd.read_csv(filepath, index_col=identifier_col)
# #     data.index = pd.DatetimeIndex(data.index, tz=tz)
# #     data.sort_index(inplace=True)
# #     return data
# #
# #
# # def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
# #     data = None
# #     for file in os.listdir(folderpath):
# #         if '.csv' not in file:
# #             continue
# #         raw = load_prices_from_csv(os.path.join(folderpath, file),
# #                                    identifier_col, tz)
# #         if data is None:
# #             data = raw
# #         else:
# #             data = pd.concat([data, raw], axis=1)
# #     return data
# #
# # def has_data_for_dates(series_or_df, first_date, last_date):
# #     """
# #     Does `series_or_df` have data on or before first_date and on or after
# #     last_date?
# #     """
# #     dts = series_or_df.index
# #     if not isinstance(dts, pd.DatetimeIndex):
# #         raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
# #     first, last = dts[[0, -1]]
# #     return (first <= first_date) and (last >= last_date)
#
# # adjustments = adjustments.reindex_axis(ADJUSTMENT_COLUMNS, axis=1)
# # row_loc = dates.get_loc(apply_date, method='bfill')
# # date_ix = np.searchsorted(dates, dividends.ex_date.values)
# # date_ix = np.searchsorted(dates, dividends.ex_date.values)
# # mask = date_ix > 0
# #
# # date_ix = date_ix[mask]
# # sids_ix = sids_ix[mask]
# # input_dates = dividends.ex_date.values[mask]
# #
# # # subtract one day to get the close on the day prior to the merger
# # previous_close = close[date_ix - 1, sids_ix]
# # input_sids = input_sids[mask]
# #
# # amount = dividends.amount.values[mask]
# # ratio = 1.0 - amount / previous_close
# # class AjustmentsWriter(object):
# #
# #     def __init__(self, engine):
# #         self.conn = engine.conncect()
# #         self.tables = sa.MetaData(bind = engine).tables
# #         self._init_declared_date()
# #         self.trading_days = Calendar(self.conn).trading_days
# #
# #     def _get_max_declared_date_from_sqlite(self, tbl):
# #         table = self.tables[tbl]
# #         sql = sa.select([table.c.sid, sa.func.max(table.c.declared_date)])
# #         sql = sql.group_by(table.c.sid)
# #         rp = self.conn.execute(sql)
# #         res = {r.sid:r.declared_date for r in rp.fetchall()}
# #         return res
# #
# #     def _init_declared_date(self):
# #         self._declared_date  = dict()
# #         for tbl in frozenset(('symbol_rights','symbol_splits')):
# #             self._declared_date[tbl] = self._get_max_declared_date_from_sqlite(tbl)
# #
# #     def request_shareBonus(self, code):
# #         url = ADJUSTMENT_URL['shareBonus']%code
# #         obj = _parse_url(url)
# #         self._download_splits_divdend(obj, code)
# #         self._download_issues(obj, code)

# # def _init(self, min_percentile, max_percentile, *args, **kwargs):
# #     self._min_percentile = min_percentile
# #     self._max_percentile = max_percentile
# #     return super(PercentileFilter, self)._init(*args, **kwargs)
# # XXX: This docstring was mostly written when the abstraction here was
# # "MultiDimensionalDataSet". It probably needs some rewriting.
# # class DataSetFamily(with_metaclass(DataSetFamilyMeta)):
# #     """
# #     Base class for Pipeline dataset families.
# #
# #     Dataset families are used to represent data where the unique identifier for
# #     a row requires more than just asset and date coordinates. A
# #     :class:`DataSetFamily` can also be thought of as a collection of
# #     :class:`~zipline.pipe.data.DataSet` objects, each of which has the same
# #     columns, domain, and ndim.
# #
# #     :class:`DataSetFamily` objects are defined with by one or more
# #     :class:`~zipline.pipe.data.Column` objects, plus one additional field:
# #     ``extra_dims``.
# #
# #     The ``extra_dims`` field defines coordinates other than asset and date that
# #     must be fixed to produce a logical timeseries. The column objects determine
# #     columns that will be shared by slices of the family.
# #
# #     ``extra_dims`` are represented as an ordered dictionary where the keys are
# #     the dimension name, and the values are a set of unique values along that
# #     dimension.
# #
# #     To work with a :class:`DataSetFamily` in a pipe expression, one must
# #     choose a specific value for each of the extra dimensions using the
# #     :meth:`~zipline.pipe.data.DataSetFamily.slice` method.
# #     For example, given a :class:`DataSetFamily`:
# #
# #     .. code-block:: python
# #
# #        class SomeDataSet(DataSetFamily):
# #            extra_dims = [
# #                ('dimension_0', {'a', 'b', 'c'}),
# #                ('dimension_1', {'d', 'e', 'f'}),
# #            ]
# #
# #            column_0 = Column(float)
# #            column_1 = Column(bool)
# #
# #     This dataset might represent a table with the following columns:
# #
# #     ::
# #
# #       sid :: int64
# #       asof_date :: datetime64[ns]
# #       timestamp :: datetime64[ns]
# #       dimension_0 :: str
# #       dimension_1 :: str
# #       column_0 :: float64
# #       column_1 :: bool
# #
# #     Here we see the implicit ``sid``, ``asof_date`` and ``timestamp`` columns
# #     as well as the extra dimensions columns.
# #
# #     This :class:`DataSetFamily` can be converted to a regular :class:`DataSet`
# #     with:
# #
# #     .. code-block:: python
# #
# #        DataSetSlice = SomeDataSet.slice(dimension_0='a', dimension_1='e')
# #
# #     This sliced dataset represents the rows from the higher dimensional dataset
# #     where ``(dimension_0 == 'a') & (dimension_1 == 'e')``.
# #     """
# #     _abstract = True  # Removed by metaclass
# #
# #     domain = GENERIC
# #     slice_ndim = 2
# #
# #     _SliceType = DataSetFamilySlice
# #
# #     @type.__call__
# #     class extra_dims(object):
# #         """OrderedDict[str, frozenset] of dimension name -> unique values
# #
# #         May be defined on subclasses as an iterable of pairs: the
# #         metaclass converts this attribute to an OrderedDict.
# #         """
# #         __isabstractmethod__ = True
# #
# #         def __get__(self, instance, owner):
# #             return []
# #
# #     @classmethod
# #     def _canonical_key(cls, args, kwargs):
# #         extra_dims = cls.extra_dims
# #         dimensions_set = set(extra_dims)
# #         if not set(kwargs) <= dimensions_set:
# #             extra = sorted(set(kwargs) - dimensions_set)
# #             raise TypeError(
# #                 '%s does not have the following %s: %s\n'
# #                 'Valid dimensions are: %s' % (
# #                     cls.__name__,
# #                     s('dimension', extra),
# #                     ', '.join(extra),
# #                     ', '.join(extra_dims),
# #                 ),
# #             )
# #
# #         if len(args) > len(extra_dims):
# #             raise TypeError(
# #                 '%s has %d extra %s but %d %s given' % (
# #                     cls.__name__,
# #                     len(extra_dims),
# #                     s('dimension', extra_dims),
# #                     len(args),
# #                     plural('was', 'were', args),
# #                 ),
# #             )
# #
# #         missing = object()
# #         coords = OrderedDict(zip(extra_dims, repeat(missing)))
# #         to_add = dict(zip(extra_dims, args))
# #         coords.update(to_add)
# #         added = set(to_add)
# #
# #         for key, value in kwargs.items():
# #             if key in added:
# #                 raise TypeError(
# #                     '%s got multiple values for dimension %r' % (
# #                         cls.__name__,
# #                         coords,
# #                     ),
# #                 )
# #             coords[key] = value
# #             added.add(key)
# #
# #         missing = {k for k, v in coords.items() if v is missing}
# #         if missing:
# #             missing = sorted(missing)
# #             raise TypeError(
# #                 'no coordinate provided to %s for the following %s: %s' % (
# #                     cls.__name__,
# #                     s('dimension', missing),
# #                     ', '.join(missing),
# #                 ),
# #             )
# #
# #         # validate that all of the provided values exist along their given
# #         # dimensions
# #         for key, value in coords.items():
# #             if value not in cls.extra_dims[key]:
# #                 raise ValueError(
# #                     '%r is not a value along the %s dimension of %s' % (
# #                         value,
# #                         key,
# #                         cls.__name__,
# #                     ),
# #                 )
# #
# #         return coords, tuple(coords.items())
# #
# #     @classmethod
# #     def _make_dataset(cls, coords):
# #         """Construct a new dataset given the coordinates.
# #         """
# #         class Slice(cls._SliceType):
# #             extra_coords = coords
# #
# #         Slice.__name__ = '%s.slice(%s)' % (
# #             cls.__name__,
# #             ', '.join('%s=%r' % item for item in coords.items()),
# #         )
# #         return Slice
# #
# #     @classmethod
# #     def slice(cls, *args, **kwargs):
# #         """Take a slice of a DataSetFamily to produce a dataset
# #         indexed by asset and date.
# #
# #         Parameters
# #         ----------
# #         *args
# #         **kwargs
# #             The coordinates to fix along each extra dimension.
# #
# #         Returns
# #         -------
# #         dataset : DataSet
# #             A regular pipe dataset indexed by asset and date.
# #
# #         Notes
# #         -----
# #         The extra dimensions coords used to produce the result are available
# #         under the ``extra_coords`` attribute.
# #         """
# #         coords, hash_key = cls._canonical_key(args, kwargs)
# #         try:
# #             return cls._slice_cache[hash_key]
# #         except KeyError:
# #             pass
# #
# #         Slice = cls._make_dataset(coords)
# #         cls._slice_cache[hash_key] = Slice
# #         return Slice
# # class DataSetFamilyMeta(abc.ABCMeta):
# #
# #     def __new__(cls, name, bases, dict_):
# #         columns = {}
# #         for k, v in dict_.items():
# #             if isinstance(v, Column):
# #                 # capture all the columns off the DataSetFamily class
# #                 # and replace them with a descriptor that will raise a helpful
# #                 # error message. The columns will get added to the BaseSlice
# #                 # for this type.
# #                 columns[k] = v
# #                 dict_[k] = _DataSetFamilyColumn(k)
# #
# #         is_abstract = dict_.pop('_abstract', False)
# #
# #         self = super(DataSetFamilyMeta, cls).__new__(
# #             cls,
# #             name,
# #             bases,
# #             dict_,
# #         )
# #
# #         if not is_abstract:
# #             self.extra_dims = extra_dims = OrderedDict([
# #                 (k, frozenset(v))
# #                 for k, v in OrderedDict(self.extra_dims).items()
# #             ])
# #             if not extra_dims:
# #                 raise ValueError(
# #                     'DataSetFamily must be defined with non-empty'
# #                     ' extra_dims, or with `_abstract = True`',
# #                 )
# #
# #             class BaseSlice(self._SliceType):
# #                 dataset_family = self
# #
# #                 ndim = self.slice_ndim
# #                 domain = self.domain
# #
# #                 locals().update(columns)
# #
# #             BaseSlice.__name__ = '%sBaseSlice' % self.__name__
# #             self._SliceType = BaseSlice
# #
# #         # each type gets a unique cache
# #         self._slice_cache = {}
# #         return self
# #
# #     def __repr__(self):
# #         return '<DataSetFamily: %r, extra_dims=%r>' % (
# #             self.__name__,
# #             list(self.extra_dims),
# #         )
# #
#
# # self.estimates = estimates[
# #     estimates[EVENT_DATE_FIELD_NAME].notnull() &
# #     estimates[FISCAL_QUARTER_FIELD_NAME].notnull() &
# #     estimates[FISCAL_YEAR_FIELD_NAME].notnull()
# #     ]
# # self.estimates[NORMALIZED_QUARTERS] = normalize_quarters(
# #     self.estimates[FISCAL_YEAR_FIELD_NAME],
# #     self.estimates[FISCAL_QUARTER_FIELD_NAME],
# # )
# #
# # self.array_overwrites_dict = {
# #     datetime64ns_dtype: Datetime641DArrayOverwrite,
# #     float64_dtype: Float641DArrayOverwrite,
# # }
# # self.scalar_overwrites_dict = {
# #     datetime64ns_dtype: Datetime64Overwrite,
# #     float64_dtype: Float64Overwrite,
# # }
# #
# # self.name_map = name_map
# # values = coerce(list, partial(np.asarray, dtype=object))
# # toolz.functoolz.flip[source]
# # Call the function call with the arguments flipped
# # NamedTemporaryFile has a visble name in file system can be retrieved from the name attribute , delete --- True means delete as file closed
# # def element_of(self, container):
# #     """
# #     Check if each element of self is an of ``container``.
# #
# #     Parameters
# #     ----------
# #     container : object
# #         An object implementing a __contains__ to call on each element of
# #         ``self``.
# #
# #     Returns
# #     -------
# #     is_contained : np.ndarray[bool]
# #         An array with the same shape as self indicating whether each
# #         element of self was an element of ``container``.
# #     """
# #     return self.map_predicate(container.__contains__)
# # functools.total_ordering(cls)
# # Given a class defining one or more rich comparison ordering methods,
# # this class decorator supplies the rest. This simplifies the effort involved in specifying all of the possible rich comparison operations:
# # The class must define one of __lt__(), __le__(), __gt__(), or __ge__(). In addition, the class should supply an __eq__() method.
# import warnings
# #
# # def _deprecated_getitem_method(name, attrs):
# #     """Create a deprecated ``__getitem__`` method that tells users to use
# #     getattr instead.
# #
# #     Parameters
# #     ----------
# #     name : str
# #         The name of the object in the warning message.
# #     attrs : iterable[str]
# #         The set of allowed attributes.
# #
# #     Returns
# #     -------
# #     __getitem__ : callable[any, str]
# #         The ``__getitem__`` method to put in the class dict.
# #     """
# #     attrs = frozenset(attrs)
# #     msg = (
# #         "'{name}[{attr!r}]' is deprecated, please use"
# #         " '{name}.{attr}' instead"
# #     )
# #
# #     def __getitem__(self, key):
# #         """``__getitem__`` is deprecated, please use attribute access instead.
# #         """
# #         warnings(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
# #         if key in attrs:
# #             return getattr(self, key)
# #         raise KeyError(key)
# #
# #     return __getitem__
#
# # @property
# # def first_trading_day(self,sid):
# #     """
# #     Returns
# #     -------
# #     dt : pd.Timestamp
# #         The first trading day (session) for which the reader can provide
# #         data.
# #     """
# #     orm = select([self.equity_basics.c.initial_date]).where(self.equity_basics.c.sid == sid)
# #     first_dt = self.conn.execute(orm).scalar()
# #     return first_dt
# #
# # @property
# # def get_last_traded_dt(self, asset):
# #     """
# #     Get the latest minute on or before ``dt`` in which ``asset`` traded.
# #
# #     If there are no trades on or before ``dt``, returns ``pd.NaT``.
# #
# #     Parameters
# #     ----------
# #     asset : zipline.asset.Asset
# #         The asset for which to get the last traded minute.
# #     dt : pd.Timestamp
# #         The minute at which to start searching for the last traded minute.
# #
# #     Returns
# #     -------
# #     last_traded : pd.Timestamp
# #         The dt of the last trade for the given asset, using the input
# #         dt as a vantage point.
# #     """
# #     orm = select([self.symbol_delist.c.delist_date]).where(self.symbol_delist.c.sid == asset)
# #     rp = self.conn.execute(orm)
# #     dead_date = rp.scalar()
# #     return dead_date
#
# # def shift_calendar(self,dt,window):
# #     window = - abs(window)
# #     index = np.searchsorted(self.all_sessions,dt)
# #     loc = index if self.all_sessions[index] == dt else index -1
# #     if loc + window < 0:
# #         raise ValueError('out of trading_calendar range')
# #     return self.all_sessions[loc + window]
#
# def cartesian(arrays, out=None):
#     """
#         参数组合 ，不同于product
#     """
#     arrays = [np.asarray(x) for x in arrays]
#     print('arrays',arrays)
#     shape = (len(x) for x in arrays)
#     print('shape',shape)
#     dtype = arrays[0].dtype
#
#     ix = np.indices(shape)
#     print('ix',ix)
#     ix = ix.reshape(len(arrays), -1).T
#     print('ix_:',ix)
#
#     if out is None:
#         out = np.empty_like(ix, dtype=dtype)
#         print('out',out.shape)
#
#     for n, arr in enumerate(arrays):
#         print('array',arrays[n])
#         print(ix[:,n])
#         out[:, n] = arrays[n][ix[:, n]]
#         print(out[:,n])
#
#     return out


# class c_test(object):

# def initialize(self):
#     pass
#
# def handle_data(self):
#     pass
#
# def before_trading_start(self):
#     pass

# class Algorithm(ABC):
#
#     def __init__(self,algo_params,data_portal):
#
#         self.algo_params = algo_params
#         self.data_portal = data_portal
#
#     def handle_data(self):
#         """
#             handle_data to run algorithm
#         """
#     @abstractmethod
#     def before_trading_start(self,dt,asset):
#         """
#             计算持仓股票的卖出信号
#         """
#
#     @abstractmethod
#     def initialzie(self,dt):
#         """
#            run algorithm on dt
#         """
#         pass
#
#
# def validate_dtype(termname, dtype, missing_value):
#     """
#     Validate a `dtype` and `missing_value` passed to Term.__new__.
#
#     Ensures that we know how to represent ``dtype``, and that missing_value
#     is specified for types without default missing values.
#
#     Returns
#     -------
#     validated_dtype, validated_missing_value : np.dtype, any
#         The dtype and missing_value to use for the new term.
#
#     Raises
#     ------
#     DTypeNotSpecified
#         When no dtype was passed to the instance, and the class doesn't
#         provide a default.
#     NotDType
#         When either the class or the instance provides a value not
#         coercible to a numpy dtype.
#     NoDefaultMissingValue
#         When dtype requires an explicit missing_value, but
#         ``missing_value`` is NotSpecified.
#     """
#     if dtype is NotSpecified:
#         raise DTypeNotSpecified(termname=termname)
#
#     try:
#         dtype = dtype_class(dtype)
#     except TypeError:
#         raise NotDType(dtype=dtype, termname=termname)
#
#     if not can_represent_dtype(dtype):
#         raise UnsupportedDType(dtype=dtype, termname=termname)
#
#     if missing_value is NotSpecified:
#         missing_value = default_missing_value_for_dtype(dtype)
#
#     try:
#         if (dtype == categorical_dtype):
#             # This check is necessary because we use object dtype for
#             # categoricals, and numpy will allow us to promote numerical
#             # values to object even though we don't support them.
#             _assert_valid_categorical_missing_value(missing_value)
#
#         # For any other type, we can check if the missing_value is safe by
#         # making an array of that value and trying to safely convert it to
#         # the desired type.
#         # 'same_kind' allows casting between things like float32 and
#         # float64, but not str and int.
#         array([missing_value]).astype(dtype=dtype, casting='same_kind')
#     except TypeError as e:
#         raise TypeError(
#             "Missing value {value!r} is not a valid choice "
#             "for term {termname} with dtype {dtype}.\n\n"
#             "Coercion attempt failed with: {error}".format(
#                 termname=termname,
#                 value=missing_value,
#                 dtype=dtype,
#                 error=e,
#             )
#         )
#
#     return dtype, missing_value


#
# def deprecated(msg=None, stacklevel=2):
#     """
#     Used to mark a function as deprecated.
#     Parameters
#     ----------
#     msg : str
#         The message to display in the deprecation warning.
#     stacklevel : int
#         How far up the stack the warning needs to go, before
#         showing the relevant calling lines.
#     Usage
#     -----
#     @deprecated(msg='function_a is deprecated! Use function_b instead.')
#     def function_a(*args, **kwargs):
#     """
#     def deprecated_dec(fn):
#         @wraps(fn)
#         def wrapper(*args, **kwargs):
#             warnings.warn(
#                 msg or "Function %s is deprecated." % fn.__name__,
#                 category=DeprecationWarning,
#                 stacklevel=stacklevel
#             )
#             return fn(*args, **kwargs)
#         return wrapper
#     return deprecated_dec
#
# def copy(self):
#     """Copy an adjusted array, deep-copying the ``data`` array.
#     """
#     if self._unvalidated:
#         raise ValueError('cannot copy unvalidated AdjustedArray')
#
#     return type(self)()


import array,binascii,struct , numpy as np,sqlalchemy as sa

# _cache = WeakValueDictionary()
#
#
# def __new__(cls,
#             engine,
#             metadata,
#             _trading_calendar):
#     identity = (engine, metadata, _trading_calendar)
#     try:
#         instance = cls._cache[identity]
#     except KeyError:
#         instance = cls._cache[identity] = super(AssetFinder, cls).__new__(cls)._initialize(*identity)
#     return instance

# if __name__ == '__main__':
#
#     engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
#                             pool_size=50,
#                             max_overflow=100,
#                             pool_timeout=-1,
#                             pool_pre_ping=True,
#                             isolation_level="READ UNCOMMITTED")
#     con = engine.connect().execution_options(isolation_level = "READ UNCOMMITTED")
#     print(con.get_execution_options())
# hstock= sa.select(self.dual_symbol.hk).\
#           where(self.dual_symbol.c.sid == sid).\
#           execute().scalar()

# #A股交易日
# trading_calendar = sa.Table(
#     'trading_calendar',
#     metadata,
#     sa.Column(
#         'trading_day',
#         sa.Text,
#         unique=True,
#         nullable=False,
#         primary_key=True,
#         index = True,
#     ),
# )
# dividend_info.append({
#     "declared_date": dividend_tuple[1],
#     "ex_date": pd.Timestamp(dividend_tuple[2], unit="s"),
#     "pay_date": pd.Timestamp(dividend_tuple[3], unit="s"),
#     "payment_sid": dividend_tuple[4],
#     "ratio": dividend_tuple[5],
#     "record_date": pd.Timestamp(dividend_tuple[6], unit="s"),
#     "sid": dividend_tuple[7]
# })


# # @preprocess(engine=coerce_string_to_eng(require_exists=False))
#
# def _generate_output_dataframe(self,data, default_cols):
#     """
#     Generates an output dataframe from the given subset of user-provided
#     data, the given column names, and the given default values.
#
#     Parameters
#     ----------
#     data : dict
#         A DataFrame, usually from an AssetData object,
#         that contains the user's input metadata for the asset type being
#         processed
#     default_cols : dict
#         A dict where the keys are the names of the columns of the desired
#         output DataFrame and the values are a function from dataframe and
#         column name to the default values to insert in the DataFrame if no user
#         data is provided
#
#     Returns
#     -------
#     DataFrame
#         A DataFrame containing all user-provided metadata, and default values
#         wherever user-provided metadata was missing
#     """
#     def _reformat_data(data):
#         _rename_cols = keyfilter(lambda x: x in equity_columns, default_cols)
#         insert_values = valmap(lambda x: data[x], _rename_cols)
#         return insert_values
#     #
#     data_subset = [_reformat_data(d) for d in data]
#     return data_subset

# def _dt_to_epoch_ns(dt_series):
#     """Convert a timeseries into an Int64Index of nanoseconds since the epoch.
#
#     Parameters
#     ----------
#     dt_series : pd.Series
#         The timeseries to convert.
#
#     Returns
#     -------
#     idx : pd.Int64Index
#         The index converted to nanoseconds since the epoch.
#     """
#     index = pd.to_datetime(dt_series.values)
#     if index.tzinfo is None:
#         index = index.tz_localize('UTC')
#     else:
#         index = index.tz_convert('UTC')
#     return index.view(np.int64)

# def split_delimited_symbol(symbol):
#     """
#     Takes in a symbol that may be delimited and splits it in to a company
#     symbol and share class symbol. Also returns the fuzzy symbol, which is the
#     symbol without any fuzzy characters at all.
#
#     Parameters
#     ----------
#     symbol : str
#         The possibly-delimited symbol to be split
#
#     Returns
#     -------
#     company_symbol : str
#         The company part of the symbol.
#     share_class_symbol : str
#         The share class part of a symbol.
#     """
#     # return blank strings for any bad fuzzy symbols, like NaN or None
#     if symbol in _delimited_symbol_default_triggers:
#         return '', ''
#
#     symbol = symbol.upper()
#
#     split_list = re.split(
#         pattern=_delimited_symbol_delimiters_regex,
#         string=symbol,
#         maxsplit=1,
#     )
#
#     # Break the list up in to its two extension, the company symbol and the
#     # share class symbol
#     company_symbol = split_list[0]
#     if len(split_list) > 1:
#         share_class_symbol = split_list[1]
#     else:
#         share_class_symbol = ''
#
#     return company_symbol, share_class_symbol
#
#
# def _check_asset_group(group):
#     row = group.sort_values('end_date').iloc[-1]
#     row.start_date = group.start_date.min()
#     row.end_date = group.end_date.max()
#     row.drop(list(symbol_columns), inplace=True)
#     return row
#
#
# def _format_range(r):
#     return (
#         str(pd.Timestamp(r.start, unit='ns')),
#         str(pd.Timestamp(r.stop, unit='ns')),
#     )
#
#
# def _check_symbol_mappings(df, exchanges, asset_exchange):
#     """Check that there are no cases where multiple symbols resolve to the same
#     asset at the same time in the same country.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The equity symbol mappings table.
#     exchanges : pd.DataFrame
#         The exchanges table.
#     asset_exchange : pd.Series
#         A series that maps sids to the exchange the asset is in.
#
#     Raises
#     ------
#     ValueError
#         Raised when there are ambiguous symbol mappings.
#     """
#     mappings = df.set_index('sid')[list(mapping_columns)].copy()
#     mappings['country_code'] = exchanges['country_code'][
#         asset_exchange.loc[df['sid']]
#     ].values
#     ambigious = {}
#
#     def check_intersections(persymbol):
#         intersections = list(intersecting_ranges(map(
#             from_tuple,
#             zip(persymbol.start_date, persymbol.end_date),
#         )))
#         if intersections:
#             data = persymbol[
#                 ['start_date', 'end_date']
#             ].astype('datetime64[ns]')
#             # indent the dataframe string, also compute this early because
#             # ``persymbol`` is a view and ``astype`` doesn't copy the index
#             # correctly in pandas 0.22
#             msg_component = '\n  '.join(str(data).splitlines())
#             ambigious[persymbol.name] = intersections, msg_component
#
#     mappings.groupby(['symbol', 'country_code']).apply(check_intersections)
#
#     if ambigious:
#         raise ValueError(
#             'Ambiguous ownership for %d symbol%s, multiple asset held the'
#             ' following symbols:\n%s' % (
#                 len(ambigious),
#                 '' if len(ambigious) == 1 else 's',
#                 '\n'.join(
#                     '%s (%s):\n  intersections: %s\n  %s' % (
#                         symbol,
#                         country_code,
#                         tuple(map(_format_range, intersections)),
#                         cs,
#                     )
#                     for (symbol, country_code), (intersections, cs) in sorted(
#                         ambigious.items(),
#                         key=first,
#                     ),
#                 ),
#             )
#         )
#
#
# def _split_symbol_mappings(df, exchanges):
#     """Split out the symbol: sid mappings from the raw data.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The dataframe with multiple rows for each symbol: sid pair.
#     exchanges : pd.DataFrame
#         The exchanges table.
#
#     Returns
#     -------
#     asset_info : pd.DataFrame
#         The asset info with one row per asset.
#     symbol_mappings : pd.DataFrame
#         The dataframe of just symbol: sid mappings. The index will be
#         the sid, then there will be three columns: symbol, start_date, and
#         end_date.
#     """
#     mappings = df[list(mapping_columns)]
#     with pd.option_context('mode.chained_assignment', None):
#         mappings['sid'] = mappings.index
#     mappings.reset_index(drop=True, inplace=True)
#
#     # take the most recent sid->exchange mapping based on end date
#     asset_exchange = df[
#         ['exchange', 'end_date']
#     ].sort_values('end_date').groupby(level=0)['exchange'].nth(-1)
#
#     _check_symbol_mappings(mappings, exchanges, asset_exchange)
#     return (
#         df.groupby(level=0).apply(_check_asset_group),
#         mappings,
#     )
#
#
# def _dt_to_epoch_ns(dt_series):
#     """Convert a timeseries into an Int64Index of nanoseconds since the epoch.
#
#     Parameters
#     ----------
#     dt_series : pd.Series
#         The timeseries to convert.
#
#     Returns
#     -------
#     idx : pd.Int64Index
#         The index converted to nanoseconds since the epoch.
#     """
#     index = pd.to_datetime(dt_series.values)
#     if index.tzinfo is None:
#         index = index.tz_localize('UTC')
#     else:
#         index = index.tz_convert('UTC')
#     return index.view(np.int64)
#
#
# def check_version_info(conn, version_table, expected_version):
#     """
#     Checks for a version value in the version table.
#
#     Parameters
#     ----------
#     conn : sa.Connection
#         The connection to use to perform the check.
#     version_table : sa.Table
#         The version table of the asset database
#     expected_version : int
#         The expected version of the asset database
#
#     Raises
#     ------
#     AssetDBVersionError
#         If the version is in the table and not equal to ASSET_DB_VERSION.
#     """
#
#     # Read the version out of the table
#     version_from_table = conn.execute(
#         sa.select((version_table.c.version,)),
#     ).scalar()
#
#     # A db without a version is considered v0
#     if version_from_table is None:
#         version_from_table = 0
#
#     # Raise an error if the versions do not match
#     if (version_from_table != expected_version):
#         raise AssetDBVersionError(db_version=version_from_table,
#                                   expected_version=expected_version)
#
#
# def write_version_info(conn, version_table, version_value):
#     """
#     Inserts the version value in to the version table.
#
#     Parameters
#     ----------
#     conn : sa.Connection
#         The connection to use to execute the insert.
#     version_table : sa.Table
#         The version table of the asset database
#     version_value : int
#         The version to write in to the database
#
#     """
#     conn.execute(sa.insert(version_table, values={'version': version_value}))
#
# Fuzzy symbol delimiters that may break up a company symbol and share class
# _delimited_symbol_delimiters_regex = re.compile(r'[./\-_]')
# _delimited_symbol_default_triggers = frozenset({np.nan, None, ''})
#
# def _default_none(df, column):
#     return None
#
# def _no_default(df, column):
#     if not df.empty:
#         raise ValueError('no default value for column %r' % column)
#
#
# # Default values for the equities DataFrame
# _equities_defaults = {
#     'symbol': _default_none,
#     'asset_name': _default_none,
#     'start_date': lambda df, col: 0,
#     # Machine limits for integer types.
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
#     'first_traded': _default_none,
#     'auto_close_date': _default_none,
#     # the full exchange name
#     'exchange': _no_default,
# }


# Default values for the root_symbols DataFrame
# _root_symbols_defaults = {
#     'sector': _default_none,
#     'description': _default_none,
#     'exchange': _default_none,
# }
#
# # Default values for the equity_supplementary_mappings DataFrame
# _equity_supplementary_mappings_defaults = {
#     'value': _default_none,
#     'field': _default_none,
#     'start_date': lambda df, col: 0,
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
# }
#
# # Default values for the equity_symbol_mappings DataFrame
# _equity_symbol_mappings_defaults = {
#     'sid': _no_default,
#     'company_symbol': _default_none,
#     'share_class_symbol': _default_none,
#     'symbol': _default_none,
#     'start_date': lambda df, col: 0,
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
# }

# MarketType

# The columns provided.
# 在ar1中但不在ar2中的已排序的唯一值
# missing_sids = np.setdiff1d(asset, self.sids)
# def compute_asset_lifetimes(frames):
#     """
#     Parameters
#     ----------
#     frames : dict[str, pd.DataFrame]
#         A dict mapping each OHLCV field to a dataframe with a row for
#         each date and a column for each sid, as passed to write().
#
#     Returns
#     -------
#     start_date_ixs : np.array[int64]
#         The index of the first date with non-nan values, for each sid.
#     end_date_ixs : np.array[int64]
#         The index of the last date with non-nan values, for each sid.
#     """
#     # Build a 2D array (dates x sids), where an entry is True if all
#     # fields are nan for the given day and sid.
#     is_null_matrix = np.logical_and.reduce(
#         [frames[field].isnull().values for field in FIELDS],
#     )
#     if not is_null_matrix.size:
#         empty = np.array([], dtype='int64')
#         return empty, empty.copy()
#
#     # Offset of the first null from the start of the input.
#     start_date_ixs = is_null_matrix.argmin(axis=0)
#     # Offset of the last null from the **end** of the input.
#     end_offsets = is_null_matrix[::-1].argmin(axis=0)
#     # Offset of the last null from the start of the input
#     end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1
#     return start_date_ixs, end_date_ixs
#
# def _make_sids():
#     asset = np.array(asset)
#     sid_selector = self.sids.searchsorted(asset)
#     #查找相同的列，invert = True
#     unknown = np.in1d(asset, self.sids, invert=True)
#     sid_selector[unknown] = -1
#     return sid_selector
#
#
# def contextmanager(f):
#     """
#     Wrapper for contextlib.contextmanager that tracks which methods of
#     PipelineHooks are contextmanagers in CONTEXT_MANAGER_METHODS.
#     """
#     PIPELINE_HOOKS_CONTEXT_MANAGERS.add(f.__name__)
#     return contextmanager(f)

# def _load_cached_data(filename, first_date, last_date, now, resource_name,
#                       environ=None):
#     if resource_name == 'benchmark':
#         def from_csv(path):
#             return pd.read_csv(
#                 path,
#                 parse_dates=[0],
#                 index_col=0,
#                 header=None,
#                 # Pass squeeze=True so that we get a series instead of a frame.
#                 squeeze=True,
#             ).tz_localize('UTC')
#     else:
#         def from_csv(path):
#             return pd.read_csv(
#                 path,
#                 parse_dates=[0],
#                 index_col=0,
#             ).tz_localize('UTC')
#
# def mk(dt):
#  if not os.path.exists(dr):
#      os.makedirs(dr)
#  os.path.join(dr, name)

# import warnings
#
# class SidView:
#
#     """
#     This class exists to temporarily support the deprecated data[sid(N)] API.
#     """
#     def __init__(self, asset, data_portal, simulation_dt_func, data_frequency):
#         """
#         Parameters
#         ---------
#         asset : Asset
#             The asset for which the instance retrieves data.
#
#         data_portal : DataPortal
#             Provider for bar pricing data.
#
#         simulation_dt_func: function
#             Function which returns the current ArkQuant time.
#             This is usually bound to a method of TradingSimulation.
#
#         data_frequency: string
#             The frequency of the bar data; i.e. whether the data is
#             'daily' or 'minute' bars
#         """
#         self.asset = asset
#         self.data_portal = data_portal
#         self.simulation_dt_func = simulation_dt_func
#         self.data_frequency = data_frequency
#
#     def __getattr__(self, column):
#         # backwards compatibility code for Q1 API
#         if column == "close_price":
#             column = "close"
#         elif column == "open_price":
#             column = "open"
#         elif column == "dt":
#             return self.dt
#         elif column == "datetime":
#             return self.datetime
#         elif column == "sid":
#             return self.sid
#
#         return self.data_portal.get_spot_value(
#             self.asset,
#             column,
#             self.simulation_dt_func(),
#             self.data_frequency
#         )
#
#     def __contains__(self, column):
#         return self.data_portal.contains(self.asset, column)
#
#     def __getitem__(self, column):
#         return self.__getattr__(column)
#
#     @property
#     def sid(self):
#         return self.asset
#
#     @property
#     def dt(self):
#         return self.datetime
#
#     @property
#     def datetime(self):
#         return self.data_portal.get_last_traded_dt(
#                 self.asset,
#                 self.simulation_dt_func(),
#                 self.data_frequency)
#
#     @property
#     def current_dt(self):
#         return self.simulation_dt_func()
#
#     def mavg(self, num_minutes):
#         self._warn_deprecated("The `mavg` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "mavg", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def stddev(self, num_minutes):
#         self._warn_deprecated("The `stddev` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "stddev", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def vwap(self, num_minutes):
#         self._warn_deprecated("The `vwap` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "vwap", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def returns(self):
#         self._warn_deprecated("The `returns` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "returns", self.simulation_dt_func(),
#             self.data_frequency
#         )
#
#     def _warn_deprecated(self, msg):
#         warnings.warn(
#             msg,
#             category=ZiplineDeprecationWarning,
#             stacklevel=1
#         )

# def _create_clock(self):
#     """
#     If the clock property is not set, then create one based on frequency.
#     """
#     trading_o_and_c = self.trading_calendar.schedule.ix[
#         self.sim_params.sessions]
#     market_closes = trading_o_and_c['market_close']
#     minutely_emission = False
#
#     if self.sim_params.data_frequency == 'minute':
#         market_opens = trading_o_and_c['market_open']
#         minutely_emission = self.sim_params.emission_rate == "minute"
#
#         # The _calendar's execution times are the minutes over which we
#         # actually want to run the clock. Typically the execution times
#         # simply adhere to the market open and close times. In the case of
#         # the futures _calendar, for example, we only want to simulate over
#         # a subset of the full 24 hour _calendar, so the execution times
#         # dictate a market open time of 6:31am US/Eastern and a close of
#         # 5:00pm US/Eastern.
#         execution_opens = \
#             self.trading_calendar.execution_time_from_open(market_opens)
#         execution_closes = \
#             self.trading_calendar.execution_time_from_close(market_closes)
#     else:
#         # in daily mode, we want to have one bar per session, timestamped
#         # as the last minute of the session.
#         execution_closes = \
#             self.trading_calendar.execution_time_from_close(market_closes)
#         execution_opens = execution_closes
#
#     # FIXME generalize these values
#     before_trading_start_minutes = days_at_time(
#         self.sim_params.sessions,
#         time(8, 45),
#         "US/Eastern"
#     )
#
#     return MinuteSimulationClock(
#         self.sim_params.sessions,
#         execution_opens,
#         execution_closes,
#         before_trading_start_minutes,
#         minute_emission=minutely_emission,
#     )

#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# import pandas as pd
#
#
# class BenchmarkSource(object):
#     def __init__(self,
#                  benchmark_asset,
#                  trading_calendar,
#                  sessions,
#                  data_portal,
#                  emission_rate="daily"):
#         self.benchmark_asset = benchmark_asset
#         self.sessions = sessions
#         self.emission_rate = emission_rate
#         self.data_portal = data_portal
#
#         if len(sessions) == 0:
#             self._precalculated_series = pd.Series()
#         elif benchmark_asset is not None:
#             self._validate_benchmark(benchmark_asset)
#             (self._precalculated_series,
#              self._daily_returns) = self._initialize_precalculated_series(
#                  benchmark_asset,
#                  trading_calendar,
#                  sessions,
#                  data_portal
#               )
#         elif benchmark_returns is not None:
#             self._daily_returns = daily_series = benchmark_returns.reindex(
#                 sessions,
#             ).fillna(0)
#
#             if self.emission_rate == "minute":
#                 # we need to take the env's benchmark returns, which are daily,
#                 # and resample them to minute
#                 minutes = trading_calendar.minutes_for_sessions_in_range(
#                     sessions[0],
#                     sessions[-1]
#                 )
#
#                 minute_series = daily_series.reindex(
#                     index=minutes,
#                     method="ffill"
#                 )
#
#                 self._precalculated_series = minute_series
#             else:
#                 self._precalculated_series = daily_series
#         else:
#             raise Exception("Must provide either benchmark_asset or "
#                             "benchmark_returns.")
#
#     def get_value(self, dt):
#         """Look up the returns for a given dt.
#
#         Parameters
#         ----------
#         dt : datetime
#             The label to look up.
#
#         Returns
#         -------
#         returns : float
#             The returns at the given dt or session.
#
#         See Also
#         --------
#         :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`
#
#         .. warning::
#
#            This method expects minute inputs if ``emission_rate == 'minute'``
#            and session labels when ``emission_rate == 'daily``.
#         """
#         return self._precalculated_series.loc[dt]
#
#     def get_range(self, start_dt, end_dt):
#         """Look up the returns for a given period.
#
#         Parameters
#         ----------
#         start_dt : datetime
#             The inclusive start label.
#         end_dt : datetime
#             The inclusive end label.
#
#         Returns
#         -------
#         returns : pd.Series
#             The series of returns.
#
#         See Also
#         --------
#         :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`
#
#         .. warning::
#
#            This method expects minute inputs if ``emission_rate == 'minute'``
#            and session labels when ``emission_rate == 'daily``.
#         """
#         return self._precalculated_series.loc[start_dt:end_dt]
#
#     def daily_returns(self, start, end=None):
#         """Returns the daily returns for the given period.
#
#         Parameters
#         ----------
#         start : datetime
#             The inclusive starting session label.
#         end : datetime, optional
#             The inclusive ending session label. If not provided, treat
#             ``start`` as a scalar key.
#
#         Returns
#         -------
#         returns : pd.Series or float
#             The returns in the given period. The index will be the trading
#             _calendar in the range [start, end]. If just ``start`` is provided,
#             return the scalar value on that day.
#         """
#         if end is None:
#             return self._daily_returns[start]
#
#         return self._daily_returns[start:end]
#
#     def _validate_benchmark(self, benchmark_asset):
#         # check if this security has a stock dividend.  if so, raise an
#         # error suggesting that the user pick a different asset to use
#         # as benchmark.
#         stock_dividends = \
#             self.data_portal.get_stock_dividends(self.benchmark_asset,
#                                                  self.sessions)
#
#         if len(stock_dividends) > 0:
#             raise InvalidBenchmarkAsset(
#                 sid=str(self.benchmark_asset),
#                 dt=stock_dividends[0]["ex_date"]
#             )
#
#         if benchmark_asset.start_date > self.sessions[0]:
#             # the asset started trading after the first ArkQuant day
#             raise BenchmarkAssetNotAvailableTooEarly(
#                 sid=str(self.benchmark_asset),
#                 dt=self.sessions[0],
#                 start_dt=benchmark_asset.start_date
#             )
#
#         if benchmark_asset.end_date < self.sessions[-1]:
#             # the asset stopped trading before the last ArkQuant day
#             raise BenchmarkAssetNotAvailableTooLate(
#                 sid=str(self.benchmark_asset),
#                 dt=self.sessions[-1],
#                 end_dt=benchmark_asset.end_date
#             )
#
#     @staticmethod
#     def _compute_daily_returns(g):
#         return (g[-1] - g[0]) / g[0]
#
#     @classmethod
#     def downsample_minute_return_series(cls,
#                                         trading_calendar,
#                                         minutely_returns):
#         sessions = trading_calendar.minute_index_to_session_labels(
#             minutely_returns.index,
#         )
#         closes = trading_calendar.session_closes_in_range(
#             sessions[0],
#             sessions[-1],
#         )
#         daily_returns = minutely_returns[closes].pct_change()
#         daily_returns.index = closes.index
#         return daily_returns.iloc[1:]
#
#     def _initialize_precalculated_series(self,
#                                          asset,
#                                          trading_calendar,
#                                          trading_days,
#                                          data_portal):
#         """
#         Internal method that pre-calculates the benchmark return series for
#         use in the ArkQuant.
#
#         Parameters
#         ----------
#         asset:  Asset to use
#
#         trading_calendar: TradingCalendar
#
#         trading_days: pd.DateTimeIndex
#
#         data_portal: DataPortal
#
#         Notes
#         -----
#         If the benchmark asset started trading after the ArkQuant start,
#         or finished trading before the ArkQuant end, exceptions are raised.
#
#         If the benchmark asset started trading the same day as the ArkQuant
#         start, the first available minute price on that day is used instead
#         of the previous close.
#
#         We use history to get an adjusted price history for each day's close,
#         as of the look-back date (the last day of the ArkQuant).  Prices are
#         fully adjusted for dividends, splits, and mergers.
#
#         Returns
#         -------
#         returns : pd.Series
#             indexed by trading day, whose values represent the %
#             change from close to close.
#         daily_returns : pd.Series
#             the partial daily returns for each minute
#         """
#         if self.emission_rate == "minute":
#             minutes = trading_calendar.minutes_for_sessions_in_range(
#                 self.sessions[0], self.sessions[-1]
#             )
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 minutes[-1],
#                 bar_count=len(minutes) + 1,
#                 frequency="1m",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             return (
#                 benchmark_series.pct_change()[1:],
#                 self.downsample_minute_return_series(
#                     trading_calendar,
#                     benchmark_series,
#                 ),
#             )
#
#         start_date = asset.start_date
#         if start_date < trading_days[0]:
#             # get the window of close prices for benchmark_asset from the
#             # last trading day of the ArkQuant, going up to one day
#             # before the ArkQuant start day (so that we can get the %
#             # change on day 1)
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 trading_days[-1],
#                 bar_count=len(trading_days) + 1,
#                 frequency="1d",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             returns = benchmark_series.pct_change()[1:]
#             return returns, returns
#         elif start_date == trading_days[0]:
#             # Attempt to handle case where stock data starts on first
#             # day, in this case use the open to close return.
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 trading_days[-1],
#                 bar_count=len(trading_days),
#                 frequency="1d",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             # get a minute history window of the first day
#             first_open = data_portal.get_spot_value(
#                 asset,
#                 'open',
#                 trading_days[0],
#                 'daily',
#             )
#             first_close = data_portal.get_spot_value(
#                 asset,
#                 'close',
#                 trading_days[0],
#                 'daily',
#             )
#
#             first_day_return = (first_close - first_open) / first_open
#
#             returns = benchmark_series.pct_change()[:]
#             returns[0] = first_day_return
#             return returns, returns
#         else:
#             raise ValueError(
#                 'cannot set benchmark to asset that does not exist during'
#                 ' the ArkQuant period (asset start date=%r)' % start_date
#             )
# from numpy import iinfo, uint32
# UINT32_MAX = iinfo(uint32).max


#
# def _cleanup_expired_assets(self, dt, position_assets):
#     """
#     Clear out any asset that have expired before starting a new sim day.
#
#     Performs two functions:
#
#     1. Finds all asset for which we have open orders and clears any
#        orders whose asset are on or after their auto_close_date.
#
#     2. Finds all asset for which we have positions and generates
#        close_position events for any asset that have reached their
#        auto_close_date.
#     """
#     algo = self.algo
#
#     def past_auto_close_date(asset):
#         acd = asset.auto_close_date
#         return acd is not None and acd <= dt
#
#     # Remove positions in any sids that have reached their auto_close date.
#     assets_to_clear = \
#         [asset for asset in position_assets if past_auto_close_date(asset)]
#     metrics_tracker = algo.metrics_tracker
#     data_portal = self.data_portal
#     for asset in assets_to_clear:
#         metrics_tracker.process_close_position(asset, dt, data_portal)
#
#     # Remove open orders for any sids that have reached their auto close
#     # date. These orders get processed immediately because otherwise they
#     # would not be processed until the first bar of the next day.
#     broker = algo.broker
#     assets_to_cancel = [
#         asset for asset in broker.open_orders
#         if past_auto_close_date(asset)
#     ]
#     for asset in assets_to_cancel:
#         broker.cancel_all_orders_for_asset(asset)
#
#     # Make a copy here so that we are not modifying the list that is being
#     # iterated over.
#     for order in copy(broker.new_orders):
#         if order.status == ORDER_STATUS.CANCELLED:
#             metrics_tracker.process_order(order)
#             broker.new_orders.remove(order)

# def _adjust_cost_basis_for_commission(self, txn_cost):
#     prev_cost = self.amount * self.cost_basis
#     new_cost = prev_cost + txn_cost
#     self.cost_basis = new_cost / self.amount
#
# def get_dividends(self, sids, trading_days):
#     """
#     splits --- divdends
#
#     Returns all the stock dividends for a specific sid that occur
#     in the given trading range.
#
#     Parameters
#     ----------
#     sid: int
#         The asset whose stock dividends should be returned.
#
#     trading_days: pd.DatetimeIndex
#         The trading range.
#
#     Returns
#     -------
#     list: A list of objects with all relevant attributes populated.
#     All timestamp fields are converted to pd.Timestamps.
#     """
#     extra = set(sids) - set(self._divdends_cache)
#     if extra:
#         for sid in extra:
#             divdends = self.adjustment_reader.load_splits_for_sid(sid)
#             self._divdends_cache[sid] = divdends
#     cache = keyfilter(lambda x: x in sids, self._splits_cache)
#     out = valmap(lambda x: x[x['pay_date'].isin(trading_days)] if x else x, cache)
#     return out
#
#
# def get_rights(self, sids, trading_days):
#     """
#     Returns all the stock dividends for a specific sid that occur
#     in the given trading range.
#
#     Parameters
#     ----------
#     sid: int
#         The asset whose stock dividends should be returned.
#
#     trading_days: pd.DatetimeIndex
#         The trading range.
#
#     Returns
#     -------
#     list: A list of objects with all relevant attributes populated.
#     All timestamp fields are converted to pd.Timestamps.
#     """
#     extra = set(sids) - set(self._rights_cache)
#     if extra:
#         for sid in extra:
#             rights = self.adjustment_reader.load_splits_for_sid(sid)
#             self._rights_cache[sid] = rights
#     #
#     cache = keyfilter(lambda x: x in sids, self._rights_cache)
#     out = valmap(lambda x: x[x['pay_date'].isin(trading_days)] if x else x, cache)
#     return out
#
# class OrderExecutionBit(object):
#     '''
#     Intended to hold information about order execution. A "bit" does not
#     determine if the order has been fully/partially executed, it just holds
#     information.
#
#     Member Attributes:
#
#       - dt: datetime (float) execution time
#       - size: how much was executed
#       - price: execution price
#       - closed: how much of the execution closed an existing postion
#       - opened: how much of the execution opened a new position
#       - openedvalue: market value of the "opened" part
#       - closedvalue: market value of the "closed" part
#       - closedcomm: commission for the "closed" part
#       - openedcomm: commission for the "opened" part
#
#       - value: market value for the entire bit size
#       - comm: commission for the entire bit execution
#       - pnl: pnl generated by this bit (if something was closed)
#
#       - psize: current open position size
#       - pprice: current open position price
#
#     '''
#
#     def __init__(self,
#                  dt=None, size=0, price=0.0,
#                  closed=0, closedvalue=0.0, closedcomm=0.0,
#                  opened=0, openedvalue=0.0, openedcomm=0.0,
#                  pnl=0.0,
#                  psize=0, pprice=0.0):
#
#         self.dt = dt
#         self.size = size
#         self.price = price
#
#         self.closed = closed
#         self.opened = opened
#         self.closedvalue = closedvalue
#         self.openedvalue = openedvalue
#         self.closedcomm = closedcomm
#         self.openedcomm = openedcomm
#
#         self.value = closedvalue + openedvalue
#         self.comm = closedcomm + openedcomm
#         self.pnl = pnl
#
#         self.psize = psize
#         self.pprice = pprice
#
#
# class OrderData(object):
#     '''
#     Holds actual order data for Creation and Execution.
#
#     In the case of Creation the request made and in the case of Execution the
#     actual outcome.
#
#     Member Attributes:
#
#       - exbits : iterable of OrderExecutionBits for this OrderData
#
#       - dt: datetime (float) creation/execution time
#       - size: requested/executed size
#       - price: execution price
#         Note: if no price is given and no pricelimite is given, the closing
#         price at the time or order creation will be used as reference
#       - pricelimit: holds pricelimit for StopLimit (which has trigger first)
#       - trailamount: absolute price distance in trailing stops
#       - trailpercent: percentage price distance in trailing stops
#
#       - value: market value for the entire bit size
#       - comm: commission for the entire bit execution
#       - pnl: pnl generated by this bit (if something was closed)
#       - margin: margin incurred by the Order (if any)
#
#       - psize: current open position size
#       - pprice: current open position price
#
#     '''
#     # According to the docs, collections.deque is thread-safe with appends at
#     # both ends, there will be no pop (nowhere) and therefore to know which the
#     # new exbits are two indices are needed. At time of cloning (__copy__) the
#     # indices can be updated to match the previous end, and the new end
#     # (len(exbits)
#     # Example: start 0, 0 -> islice(exbits, 0, 0) -> []
#     # One added -> copy -> updated 0, 1 -> islice(exbits, 0, 1) -> [1 elem]
#     # Other added -> copy -> updated 1, 2 -> islice(exbits, 1, 2) -> [1 elem]
#     # "add" and "__copy__" happen always in the same thread (with all current
#     # implementations) and therefore no append will happen during a copy and
#     # the len of the exbits can be queried with no concerns about another
#     # thread making an append and with no need for a lock
#
#     def __init__(self, dt=None, size=0, price=0.0, pricelimit=0.0, remsize=0,
#                  pclose=0.0, trailamount=0.0, trailpercent=0.0):
#
#         self.pclose = pclose
#         self.exbits = collections.deque()  # for historical purposes
#         self.p1, self.p2 = 0, 0  # indices to pending notifications
#
#         self.dt = dt
#         self.size = size
#         self.remsize = remsize
#         self.price = price
#         self.pricelimit = pricelimit
#         self.trailamount = trailamount
#         self.trailpercent = trailpercent
#
#         if not pricelimit:
#             # if no pricelimit is given, use the given price
#             self.pricelimit = self.price
#
#         if pricelimit and not price:
#             # price must always be set if pricelimit is set ...
#             self.price = pricelimit
#
#         self.plimit = pricelimit
#
#         self.value = 0.0
#         self.comm = 0.0
#         self.margin = None
#         self.pnl = 0.0
#
#         self.psize = 0
#         self.pprice = 0
#
#     def _getplimit(self):
#         return self._plimit
#
#     def _setplimit(self, val):
#         self._plimit = val
#
#     plimit = property(_getplimit, _setplimit)
#
#     def __len__(self):
#         return len(self.exbits)
#
#     def __getitem__(self, key):
#         return self.exbits[key]
#
#     def add(self, dt, size, price,
#             closed=0, closedvalue=0.0, closedcomm=0.0,
#             opened=0, openedvalue=0.0, openedcomm=0.0,
#             pnl=0.0,
#             psize=0, pprice=0.0):
#
#         self.addbit(
#             OrderExecutionBit(dt, size, price,
#                               closed, closedvalue, closedcomm,
#                               opened, openedvalue, openedcomm, pnl,
#                               psize, pprice))
#
#     def addbit(self, exbit):
#         # Stores an ExecutionBit and recalculates own values from ExBit
#         self.exbits.append(exbit)
#
#         self.remsize -= exbit.size
#
#         self.dt = exbit.dt
#         oldvalue = self.size * self.price
#         newvalue = exbit.size * exbit.price
#         self.size += exbit.size
#         self.price = (oldvalue + newvalue) / self.size
#         self.value += exbit.value
#         self.comm += exbit.comm
#         self.pnl += exbit.pnl
#         self.psize = exbit.psize
#         self.pprice = exbit.pprice
#
#     def getpending(self):
#         return list(self.iterpending())
#
#     def iterpending(self):
#         return itertools.islice(self.exbits, self.p1, self.p2)
#
#     def markpending(self):
#         # rebuild the indices to mark which exbits are pending in clone
#         self.p1, self.p2 = self.p2, len(self.exbits)
#
#     def clone(self):
#         obj = copy(self)
#         obj.markpending()
#         return obj
#
#
# class OrderBase(with_metaclass(MetaParams, object)):
#     params = (
#         ('owner', None), ('data', None),
#         ('size', None), ('price', None), ('pricelimit', None),
#         ('exectype', None), ('valid', None), ('tradeid', 0), ('oco', None),
#         ('trailamount', None), ('trailpercent', None),
#         ('parent', None), ('transmit', True),
#         ('simulated', False),
#         # To support historical order evaluation
#         ('histnotify', False),
#     )
#
#     DAY = datetime.timedelta()  # constant for DAY order identification
#
#     # Time Restrictions for orders
#     T_Close, T_Day, T_Date, T_None = range(4)
#
#     # Volume Restrictions for orders
#     V_None = range(1)
#
#     (Market, Close, Limit, Stop, StopLimit, StopTrail, StopTrailLimit,
#      Historical) = range(8)
#     ExecTypes = ['Market', 'Close', 'Limit', 'Stop', 'StopLimit', 'StopTrail',
#                  'StopTrailLimit', 'Historical']
#
#     OrdTypes = ['Buy', 'Sell']
#     Buy, Sell = range(2)
#
#     Created, Submitted, Accepted, Partial, Completed, \
#         Canceled, Expired, Margin, Rejected = range(9)
#
#     Cancelled = Canceled  # alias
#
#     Status = [
#         'Created', 'Submitted', 'Accepted', 'Partial', 'Completed',
#         'Canceled', 'Expired', 'Margin', 'Rejected',
#     ]
#
#     refbasis = itertools.count(1)  # for a unique identifier per order
#
#     def _getplimit(self):
#         return self._plimit
#
#     def _setplimit(self, val):
#         self._plimit = val
#
#     plimit = property(_getplimit, _setplimit)
#
#     def __getattr__(self, name):
#         # Return attr from params if not found in order
#         return getattr(self.params, name)
#
#     def __setattribute__(self, name, value):
#         if hasattr(self.params, name):
#             setattr(self.params, name, value)
#         else:
#             super(Order, self).__setattribute__(name, value)
#
#     def __str__(self):
#         tojoin = list()
#         tojoin.append('Ref: {}'.format(self.ref))
#         tojoin.append('OrdType: {}'.format(self.ordtype))
#         tojoin.append('OrdType: {}'.format(self.ordtypename()))
#         tojoin.append('Status: {}'.format(self.status))
#         tojoin.append('Status: {}'.format(self.getstatusname()))
#         tojoin.append('Size: {}'.format(self.size))
#         tojoin.append('Price: {}'.format(self.price))
#         tojoin.append('Price Limit: {}'.format(self.pricelimit))
#         tojoin.append('TrailAmount: {}'.format(self.trailamount))
#         tojoin.append('TrailPercent: {}'.format(self.trailpercent))
#         tojoin.append('ExecType: {}'.format(self.exectype))
#         tojoin.append('ExecType: {}'.format(self.getordername()))
#         tojoin.append('CommInfo: {}'.format(self.comminfo))
#         tojoin.append('End of Session: {}'.format(self.dteos))
#         tojoin.append('Info: {}'.format(self.info))
#         tojoin.append('Broker: {}'.format(self.broker))
#         tojoin.append('Alive: {}'.format(self.alive()))
#
#         return '\n'.join(tojoin)
#
#     def __init__(self):
#         self.ref = next(self.refbasis)
#         self.broker = None
#         self.info = AutoOrderedDict()
#         self.comminfo = None
#         self.triggered = False
#
#         self._active = self.parent is None
#         self.status = Order.Created
#
#         self.plimit = self.p.pricelimit  # alias via property
#
#         if self.exectype is None:
#             self.exectype = Order.Market
#
#         if not self.isbuy():
#             self.size = -self.size
#
#         # Set a reference price if price is not set using
#         # the close price
#         pclose = self.data.close[0] if not self.simulated else self.price
#         if not self.price and not self.pricelimit:
#             price = pclose
#         else:
#             price = self.price
#
#         dcreated = self.data.datetime[0] if not self.p.simulated else 0.0
#         self.created = OrderData(dt=dcreated,
#                                  size=self.size,
#                                  price=price,
#                                  pricelimit=self.pricelimit,
#                                  pclose=pclose,
#                                  trailamount=self.trailamount,
#                                  trailpercent=self.trailpercent)
#
#         # Adjust price in case a trailing limit is wished
#         if self.exectype in [Order.StopTrail, Order.StopTrailLimit]:
#             self._limitoffset = self.created.price - self.created.pricelimit
#             price = self.created.price
#             self.created.price = float('inf' * self.isbuy() or '-inf')
#             self.trailadjust(price)
#         else:
#             self._limitoffset = 0.0
#
#         self.executed = OrderData(remsize=self.size)
#         self.position = 0
#
#         if isinstance(self.valid, datetime.date):
#             # comparison will later be done against the raw datetime[0] value
#             self.valid = self.data.date2num(self.valid)
#         elif isinstance(self.valid, datetime.timedelta):
#             # offset with regards to now ... get utcnow + offset
#             # when reading with date2num ... it will be automatically localized
#             if self.valid == self.DAY:
#                 valid = datetime.datetime.combine(
#                     self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
#             else:
#                 valid = self.data.datetime.datetime() + self.valid
#
#             self.valid = self.data.date2num(valid)
#
#         elif self.valid is not None:
#             if not self.valid:  # avoid comparing None and 0
#                 valid = datetime.datetime.combine(
#                     self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
#             else:  # assume float
#                 valid = self.data.datetime[0] + self.valid
#
#         if not self.p.simulated:
#             # provisional end-of-session
#             # get next session end
#             dtime = self.data.datetime.datetime(0)
#             session = self.data.p.sessionend
#             dteos = dtime.replace(hour=session.hour, minute=session.minute,
#                                   second=session.second,
#                                   microsecond=session.microsecond)
#
#             if dteos < dtime:
#                 # eos before current time ... no ... must be at least next day
#                 dteos += datetime.timedelta(days=1)
#
#             self.dteos = self.data.date2num(dteos)
#         else:
#             self.dteos = 0.0
#
#     def clone(self):
#         # status, triggered and executed are the only moving parts in order
#         # status and triggered are covered by copy
#         # executed has to be replaced with an intelligent clone of itself
#         obj = copy(self)
#         obj.executed = self.executed.clone()
#         return obj  # status could change in next to completed
#
#     def getstatusname(self, status=None):
#         '''Returns the name for a given status or the one of the order'''
#         return self.Status[self.status if status is None else status]
#
#     def getordername(self, exectype=None):
#         '''Returns the name for a given exectype or the one of the order'''
#         return self.ExecTypes[self.exectype if exectype is None else exectype]
#
#     @classmethod
#     def ExecType(cls, exectype):
#         return getattr(cls, exectype)
#
#     def ordtypename(self, ordtype=None):
#         '''Returns the name for a given ordtype or the one of the order'''
#         return self.OrdTypes[self.ordtype if ordtype is None else ordtype]
#
#     def active(self):
#         return self._active
#
#     def activate(self):
#         self._active = True
#
#     def alive(self):
#         '''Returns True if the order is in a status in which it can still be
#         executed
#         '''
#         return self.status in [Order.Created, Order.Submitted,
#                                Order.Partial, Order.Accepted]
#
#     def addcomminfo(self, comminfo):
#         '''Stores a CommInfo scheme associated with the asset'''
#         self.comminfo = comminfo
#
#     def addinfo(self, **kwargs):
#         '''Add the keys, values of kwargs to the internal info dictionary to
#         hold custom information in the order
#         '''
#         for key, val in iteritems(kwargs):
#             self.info[key] = val
#
#     def __eq__(self, other):
#         return other is not None and self.ref == other.ref
#
#     def __ne__(self, other):
#         return self.ref != other.ref
#
#     def isbuy(self):
#         '''Returns True if the order is a Buy order'''
#         return self.ordtype == self.Buy
#
#     def issell(self):
#         '''Returns True if the order is a Sell order'''
#         return self.ordtype == self.Sell
#
#     def setposition(self, position):
#         '''Receives the current position for the asset and stotres it'''
#         self.position = position
#
#     def submit(self, broker=None):
#         '''Marks an order as submitted and stores the broker to which it was
#         submitted'''
#         self.status = Order.Submitted
#         self.broker = broker
#         self.plen = len(self.data)
#
#     def accept(self, broker=None):
#         '''Marks an order as accepted'''
#         self.status = Order.Accepted
#         self.broker = broker
#
#     def brokerstatus(self):
#         '''Tries to retrieve the status from the broker in which the order is.
#
#         Defaults to last known status if no broker is associated'''
#         if self.broker:
#             return self.broker.orderstatus(self)
#
#         return self.status
#
#     def reject(self, broker=None):
#         '''Marks an order as rejected'''
#         if self.status == Order.Rejected:
#             return False
#
#         self.status = Order.Rejected
#         self.executed.dt = self.data.datetime[0]
#         self.broker = broker
#         return True
#
#     def cancel(self):
#         '''Marks an order as cancelled'''
#         self.status = Order.Canceled
#         self.executed.dt = self.data.datetime[0]
#
#     def margin(self):
#         '''Marks an order as having met a margin call'''
#         self.status = Order.Margin
#         self.executed.dt = self.data.datetime[0]
#
#     def completed(self):
#         '''Marks an order as completely filled'''
#         self.status = self.Completed
#
#     def partial(self):
#         '''Marks an order as partially filled'''
#         self.status = self.Partial
#
#     def execute(self, dt, size, price,
#                 closed, closedvalue, closedcomm,
#                 opened, openedvalue, openedcomm,
#                 margin, pnl,
#                 psize, pprice):
#
#         '''Receives data execution input and stores it'''
#         if not size:
#             return
#
#         self.executed.add(dt, size, price,
#                           closed, closedvalue, closedcomm,
#                           opened, openedvalue, openedcomm,
#                           pnl, psize, pprice)
#
#         self.executed.margin = margin
#
#     def expire(self):
#         '''Marks an order as expired. Returns True if it worked'''
#         self.status = self.Expired
#         return True
#
#     def trailadjust(self, price):
#         pass  # generic interface
#
#
# class Order(OrderBase):
#     '''
#     Class which holds creation/execution data and type of oder.
#
#     The order may have the following status:
#
#       - Submitted: sent to the broker and awaiting confirmation
#       - Accepted: accepted by the broker
#       - Partial: partially executed
#       - Completed: fully exexcuted
#       - Canceled/Cancelled: canceled by the user
#       - Expired: expired
#       - Margin: not enough cash to execute the order.
#       - Rejected: Rejected by the broker
#
#         This can happen during order submission (and therefore the order will
#         not reach the Accepted status) or before execution with each new bar
#         price because cash has been drawn by other sources (future-like
#         instruments may have reduced the cash or orders orders may have been
#         executed)
#
#     Member Attributes:
#
#       - ref: unique order identifier
#       - created: OrderData holding creation data
#       - executed: OrderData holding execution data
#
#       - info: custom information passed over method :func:`addinfo`. It is kept
#         in the form of an OrderedDict which has been subclassed, so that keys
#         can also be specified using '.' notation
#
#     User Methods:
#
#       - isbuy(): returns bool indicating if the order buys
#       - issell(): returns bool indicating if the order sells
#       - alive(): returns bool if order is in status Partial or Accepted
#     '''
#
#     def execute(self, dt, size, price,
#                 closed, closedvalue, closedcomm,
#                 opened, openedvalue, openedcomm,
#                 margin, pnl,
#                 psize, pprice):
#
#         super(Order, self).execute(dt, size, price,
#                                    closed, closedvalue, closedcomm,
#                                    opened, openedvalue, openedcomm,
#                                    margin, pnl, psize, pprice)
#
#         if self.executed.remsize:
#             self.status = Order.Partial
#         else:
#             self.status = Order.Completed
#
#         # self.comminfo = None
#
#     def expire(self):
#         if self.exectype == Order.Market:
#             return False  # will be executed yes or yes
#
#         if self.valid and self.data.datetime[0] > self.valid:
#             self.status = Order.Expired
#             self.executed.dt = self.data.datetime[0]
#             return True
#
#         return False
#
#     def trailadjust(self, price):
#         if self.trailamount:
#             pamount = self.trailamount
#         elif self.trailpercent:
#             pamount = price * self.trailpercent
#         else:
#             pamount = 0.0
#
#         # Stop sell is below (-), stop buy is above, move only if needed
#         if self.isbuy():
#             price += pamount
#             if price < self.created.price:
#                 self.created.price = price
#                 if self.exectype == Order.StopTrailLimit:
#                     self.created.pricelimit = price - self._limitoffset
#         else:
#             price -= pamount
#             if price > self.created.price:
#                 self.created.price = price
#                 if self.exectype == Order.StopTrailLimit:
#                     # limitoffset is negative when pricelimit was greater
#                     # the - allows increasing the price limit if stop increases
#                     self.created.pricelimit = price - self._limitoffset
#
#
# class BuyOrder(Order):
#     ordtype = Order.Buy
#
#
# class StopBuyOrder(BuyOrder):
#     pass
#
#
# class StopLimitBuyOrder(BuyOrder):
#     pass
#
#
# class SellOrder(Order):
#     ordtype = Order.Sell
#
#
# class StopSellOrder(SellOrder):
#     pass
#
#
# class StopLimitSellOrder(SellOrder):
#     pass

# class TickerOrder(Order):
#     # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
#     # Order objects and we keep them all in memory, so it's worthwhile trying
#     # to cut down on the memory footprint of this object.
#     """
#         Parameters
#         ----------
#         asset : AssetEvent
#             The asset that this order is for.
#         amount : int
#             The amount of shares to order. If ``amount`` is positive, this is
#             the number of shares to buy or cover. If ``amount`` is negative,
#             this is the number of shares to sell or short.
#         dt : str, optional
#             The date created order.
#
#         市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
#
#     """
#     __slot__ = ['asset','_created_dt','capital']
#
#     def __init__(self,asset,ticker,capital):
#         self.asset = asset
#         self._created_dt = ticker
#         self.order_capital = capital
#         self.direction = math.copysign(1,capital)
#         self.filled = 0.0
#         self.broker_order_id = self.make_id()
#         self.order_type = StyleType.BOC
#
#     def check_trigger(self,dts):
#         if dts >= self._created_dt:
#             return True
#         return False
# def fulfill(self, data, iterator):
#     # 设定价格限制 , iterator里面的对象为第一个为price
#     bottom = data.pre['close'] * (1 - self._style.get_stop_price)
#     upper = data.pre['close'] * (1 + self._style.get_limit_price)
#     # 过滤
#     _iter = [item for item in iterator if bottom < item[0] < upper]
#     return _iter

# class MarketImpact(SlippageModel):
#     """
#         基于成交量进行对市场的影响进行测算
#     """
#     def __init__(self,func = np.exp):
#         self.adjust_func = func
#
#     def calculate_slippage_factor(self,target,volume):
#         psi = target / volume.mean()
#         factor = self.adjust_func(psi)
#         return factor
# from types import MappingProxyType as mappingproxy
# 返回一个动态映射视图

# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def load_portfolio_risk_factors(filepath_prefix=None, start=None, end=None):
#     """
#     Load risk factors Mkt-Rf, SMB, HML, Rf, and UMD.
#     Data is stored in HDF5 file. If the data is more than 2
#     days old, redownload from Dartmouth.
#     Returns
#     -------
#     five_factors : pd.DataFrame
#         Risk factors timeseries.
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     start = get_utc_timestamp(start)
#     end = get_utc_timestamp(end)
#
#     if filepath_prefix is None:
#         filepath = data_path('factors.csv')
#     else:
#         filepath = filepath_prefix
#
#     five_factors = get_returns_cached(filepath, get_fama_french, end)
#
#     return five_factors.loc[start:end]
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_treasury_yield(start=None, end=None, period='3MO'):
#     """
#     Load treasury yields from FRED.
#
#     Parameters
#     ----------
#     start : date, optional
#         Earliest date to fetch data for.
#         Defaults to earliest date available.
#     end : date, optional
#         Latest date to fetch data for.
#         Defaults to latest date available.
#     period : {'1MO', '3MO', '6MO', 1', '5', '10'}, optional
#         Which maturity to use.
#     Returns
#     -------
#     pd.Series
#         Annual treasury yield for every day.
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     treasury = web.DataReader("DGS3{}".format(period), "fred",
#                               start, end)
#
#     treasury = treasury.ffill()
#
#     return treasury
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_symbol_returns_from_yahoo(symbol, start=None, end=None):
#     """
#     Wrapper for pandas.io.data.get_data_yahoo().
#     Retrieves prices for symbol from yahoo and computes returns
#     based on adjusted closing prices.
#
#     Parameters
#     ----------
#     symbol : str
#         Symbol name to load, e.g. 'SPY'
#     start : pandas.Timestamp compatible, optional
#         Start date of time period to retrieve
#     end : pandas.Timestamp compatible, optional
#         End date of time period to retrieve
#
#     Returns
#     -------
#     pandas.DataFrame
#         Returns of symbol in requested period.
#     """
#
#     try:
#         px = web.get_data_yahoo(symbol, start=start, end=end)
#         px['date'] = pd.to_datetime(px['date'])
#         px.set_index('date', drop=False, inplace=True)
#         rets = px[['adjclose']].pct_change().dropna()
#     except Exception as e:
#         warnings.warn(
#             'Yahoo Finance read failed: {}, falling back to Google'.format(e),
#             UserWarning)
#         px = web.get_data_google(symbol, start=start, end=end)
#         rets = px[['Close']].pct_change().dropna()
#
#     rets.index = rets.index.tz_localize("UTC")
#     rets.columns = [symbol]
#     return rets
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def default_returns_func(symbol, start=None, end=None):
#     """
#     Gets returns for a symbol.
#     Queries Yahoo Finance. Attempts to cache SPY.
#
#     Parameters
#     ----------
#     symbol : str
#         Ticker symbol, e.g. APPL.
#     start : date, optional
#         Earliest date to fetch data for.
#         Defaults to earliest date available.
#     end : date, optional
#         Latest date to fetch data for.
#         Defaults to latest date available.
#
#     Returns
#     -------
#     pd.Series
#         Daily returns for the symbol.
#          - See full explanation in tears.create_full_tear_sheet (returns).
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     start = get_utc_timestamp(start)
#     end = get_utc_timestamp(end)
#
#     if symbol == 'SPY':
#         filepath = data_path('spy.csv')
#         rets = get_returns_cached(filepath,
#                                   get_symbol_returns_from_yahoo,
#                                   end,
#                                   symbol='SPY',
#                                   start='1/1/1970',
#                                   end=datetime.now())
#         rets = rets[start:end]
#     else:
#         rets = get_symbol_returns_from_yahoo(symbol, start=start, end=end)
#
#     return rets[symbol]
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_fama_french():
#     """
#     Retrieve Fama-French factors via pandas-datareader
#     Returns
#     -------
#     pandas.DataFrame
#         Percent change of Fama-French factors
#     """
#
#     start = '1/1/1970'
#     research_factors = web.DataReader('F-F_Research_Data_Factors_daily',
#                                       'famafrench', start=start)[0]
#     momentum_factor = web.DataReader('F-F_Momentum_Factor_daily',
#                                      'famafrench', start=start)[0]
#     five_factors = research_factors.join(momentum_factor).dropna()
#     five_factors /= 100.
#     five_factors.index = five_factors.index.tz_localize('utc')
#
#     five_factors.columns = five_factors.columns.str.strip()
#
#     return five_factors
# try:
#     # fast versions
#     import bottleneck as bn
#
#     def _wrap_function(f):
#         @wraps(f)
#         def wrapped(*args, **kwargs):
#             out = kwargs.pop('out', None)
#             data = f(*args, **kwargs)
#             if out is None:
#                 out = data
#             else:
#                 out[()] = data
#
#             return out
#
#         return wrapped
#
#     nanmean = _wrap_function(bn.nanmean)
#     nanstd = _wrap_function(bn.nanstd)
#     nansum = _wrap_function(bn.nansum)
#     nanmax = _wrap_function(bn.nanmax)
#     nanmin = _wrap_function(bn.nanmin)
#     nanargmax = _wrap_function(bn.nanargmax)
#     nanargmin = _wrap_function(bn.nanargmin)
# except ImportError:
#     # slower numpy
#     nanmean = np.nanmean
#     nanstd = np.nanstd
#     nansum = np.nansum
#     nanmax = np.nanmax
#     nanmin = np.nanmin
#     nanargmax = np.nanargmax
#     nanargmin = np.nanargmin
#
#
# try:
#     from pandas_datareader import data as web
# except ImportError:
#     msg = ("Unable to import pandas_datareader. Suppressing import error and "
#            "continuing. All data reading functionality will raise errors; but "
#            "has been deprecated and will be removed in a later version.")
#     warnings.warn(msg)
# from .deprecate import deprecated
#
# DATAREADER_DEPRECATION_WARNING = \
#         ("Yahoo and Google Finance have suffered large API breaks with no "
#          "stable replacement. As a result, any data reading functionality "
#          "in empyrical has been deprecated and will be removed in a future "
#          "version. See README.md for more details: "
#          "\n\n"
#          "\thttps://github.com/quantopian/pyfolio/blob/master/README.md")

# from __future__ import division
# from multipledispatch import dispatch
# from .compat import PY2
# import numpy as np
#
# if PY2:
#     int_t = (int, long, np.int64)
# else:
#     int_t = (int, np.int64)
# from __future__ import division

# @property
# def default(self):
#     return self._default()
#
# def _default(self,dt):
#     """
#         a. 剔除停盘
#         b. 剔除上市不足一个月的 --- 次新股波动性太大
#         c. 剔除进入退市整理期的30个交易日
#     """
#     active_assets = self.asset_finder.was_active(dt)
#     sdate = self.trading_calendar._roll_forward(dt,StableEPeriod)
#     edate = self.trading_calendar._roll_forward(dt, -EnsurePeriod)
#     stable_alive = self.asset_finder.lifetime([sdate,edate])
#     default_assets = set(active_assets) & set(stable_alive)
#     return default_assets

# @staticmethod
# def _execution_open_and_close(_calendar, session):
#     open_, close = _calendar.open_and_close_for_session(session)
#     execution_open = _calendar.execution_time_from_open(open_)
#     execution_close = _calendar.execution_time_from_close(close)
# cal = self._trading_calendar
# self._market_open, self._market_close = self._execution_open_and_close(
#     cal,
#     session_label,
# )
# if self.emission_rate == 'daily':
#     # this method is called for both minutely and daily emissions, but
#     # this chunk of code here only applies for daily emissions. (since
#     # it's done every minute, elsewhere, for minutely emission).
#     self.sync_last_sale_prices(dt, data_portal)
# session_ix = self._session_count
# # increment the day counter before we move markers forward.
# self._session_count += 1
# self._total_session_count = len(sessions)
# self._session_count = 0
# self.emission_rate = emission_rate
# emission_rate = 'daily',
# import logging
# logging.info(
#     'Simulated {} trading days\n'
#     'first open: {}\n'
#     'last close: {}',
#     self._session_count,
#     self._trading_calendar.session_open(self._first_session),
#     self._trading_calendar.session_close(self._last_session),
# )


# Copied from Position and renamed.  This is used to handle cases where a user
# does something like `context.portfolio.positions[100]` instead of
# `context.portfolio.positions[sid(100)]`.
# class _DeprecatedSidLookupPosition(object):
#     def __init__(self, sid):
#         self.sid = sid
#         self.amount = 0
#         self.cost_basis = 0.0  # per share
#         self.last_sale_price = 0.0
#         self.last_sale_date = None
#
#     def __repr__(self):
#         return "_DeprecatedSidLookupPosition({0})".format(self.__dict__)
#
#     # If you are adding new attributes, don't update this set. This method
#     # is deprecated to normal attribute access so we don't want to encourage
#     # new usages.
#     __getitem__ = _deprecated_getitem_method(
#         'position', {
#             'sid',
#             'amount',
#             'cost_basis',
#             'last_sale_price',
#             'last_sale_date',
#         },
#     )
#
#
# class Positions(dict):
#     """A dict-like object containing the algorithm's current positions.
#     """
#
#     def __missing__(self, key):
#         if isinstance(key, Asset):
#             return Position(InnerPosition(key))
#         elif isinstance(key, int):
#             warnings.warn("Referencing positions by integer is deprecated."
#                  " Use an asset instead.")
#         else:
#             warnings.warn("Position lookup expected a value of type Asset but got {0}"
#                  " instead.".format(type(key).__name__))
#
#         return _DeprecatedSidLookupPosition(key)

# We don't have a datetime for the current snapshot until we
# receive a message.
# This object is the way that user algorithms interact with OHLCV data,
# fetcher data, and some API methods like `data.can_trade`.
# self.current_data = self._create_bar_data()

# #获取日数据，封装为一个API(fetch process flush other api)
# def _create_bar_data(self):
#     return BarData(
#         data_portal=self.data_portal,
#         data_frequency=self.sim_params.data_frequency,
#         trading_calendar=self.algo.trading_calendar,
#         restrictions=self.restrictions,
#     )
# if isinstance(asset, Asset):
#     return False
# return pd.Series(index=pd.Index(asset), data=False)
# if isinstance(asset, Asset):
#     return asset in self._restricted_set
# return pd.Series(
#     index=pd.Index(asset),
#     # list 内置的__contains__ 方法
#     # data=vectorized_is_element(asset, self._restricted_set)
#     data = np.vectorize(self._restricted_set.__contains__,otypes = [bool])(asset)
# )
# def is_restricted(self, asset, dt):
#     if isinstance(asset, Asset):
#         return any(
#             r.is_restricted(asset, dt) for r in self.sub_restrictions
#         )
#
#     return reduce(
#         operator.or_,
#         (r.is_restricted(asset, dt) for r in self.sub_restrictions)
#     )


#
# # exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# # code = compile(self.algoscript, algo_filename, 'exec')
# # exec_(code, self.namespace)
# #
# # # dict get参数可以为方法或者默认参数
# # self._initialize = self.namespace.get('initialize', noop)
# # self._handle_data = self.namespace.get('handle_data', noop)
# # self._before_trading_start = self.namespace.get(
# #     'before_trading_start',
# # )

# class BarData:
#     """
#     Provides methods for accessing minutely and daily price/volume data from
#     Algorithm API functions.
#
#     Also provides utility methods to determine if an asset is alive, and if it
#     has recent trade data.
#
#     An instance of this object is passed as ``data`` to
#     :func:`~zipline.api.handle_data` and
#     :func:`~zipline.api.before_trading_start`.
#
#     Parameters
#     ----------
#     data_portal : DataPortal
#         Provider for bar pricing data.
#     data_frequency : {'minute', 'daily'}
#         The frequency of the bar data; i.e. whether the data is
#         daily or minute bars
#     restrictions : zipline.finance.asset_restrictions.Restrictions
#         Object that combines and returns restricted list information from
#         multiple sources
#     """
#
#     def __init__(self, data_portal, data_frequency,
#                  trading_calendar, restrictions):
#         self.data_portal = data_portal
#         self.data_frequency = data_frequency
#         self._trading_calendar = trading_calendar
#         self._is_restricted = restrictions.is_restricted
#
#     def get_current_ticker(self,asset,fields):
#         """
#         Returns the "current" value of the given fields for the given asset
#         at the current ArkQuant time.
#         :param asset: asset_type
#         :param fields: OHLCTV
#         :return: dict asset -> ticker
#         intended to return current ticker
#         """
#         cur = {}
#         for asset in asset:
#             ticker = self.data_portal.get_current(asset)
#             cur[asset] = ticker.loc[:,fields]
#         return cur
#
#     def history(self, asset, end_dt,bar_count, fields,frequency):
#         """
#         Returns a trailing window of length ``bar_count`` containing data for
#         the given asset, fields, and frequency.
#
#         Returned data is adjusted for splits, dividends, and mergers as of the
#         current ArkQuant time.
#
#         The semantics for missing data are identical to the ones described in
#         the notes for :meth:`current`.
#
#         Parameters
#         ----------
#         asset: zipline.asset.Asset or iterable of zipline.asset.Asset
#             The asset(s) for which data is requested.
#         fields: string or iterable of string.
#             Requested data field(s). Valid field names are: "price",
#             "last_traded", "open", "high", "low", "close", and "volume".
#         bar_count: int
#             Number of data observations requested.
#         frequency: str
#             String indicating whether to load daily or minutely data
#             observations. Pass '1m' for minutely data, '1d' for daily data.
#
#         Returns
#         -------
#         history : pd.Series or pd.DataFrame or pd.Panel
#             See notes below.
#
#         Notes
#         ------
#         returned panel has:
#         items: fields
#         major axis: dt
#         minor axis: asset
#         return pd.Panel(df_dict)
#         """
#         sliding_window = self.data_portal.get_history_window(asset,
#                                                              end_dt,
#                                                              bar_count,
#                                                              fields,
#                                                              frequency)
#         return sliding_window
#
#     def window_data(self,asset,end_dt,bar_count,fields,frequency):
#         window_array = self.data_portal.get_window_data(asset,
#                                                         end_dt,
#                                                         bar_count,
#                                                         fields,
#                                                         frequency)
#         return window_array
# ALLOWED_READ_CSV_KWARGS = {
#     'sep',
#     'dialect',
#     'doublequote',
#     'escapechar',
#     'quotechar',
#     'quoting',
#     'skipinitialspace',
#     'lineterminator',
#     'header',
#     'index_col',
#     'names',
#     'prefix',
#     'skiprows',
#     'skipfooter',
#     'skip_footer',
#     'na_values',
#     'true_values',
#     'false_values',
#     'delimiter',
#     'converters',
#     'dtype',
#     'delim_whitespace',
#     'as_recarray',
#     'na_filter',
#     'compact_ints',
#     'use_unsigned',
#     'buffer_lines',
#     'warn_bad_lines',
#     'error_bad_lines',
#     'keep_default_na',
#     'thousands',
#     'comment',
#     'decimal',
#     'keep_date_col',
#     'nrows',
#     'chunksize',
#     'encoding',
#     'usecols'
# }


# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#         for limit_price ,this means preferring to round down on buys and preferring to round up on sells.
#         for stop_price ,reverse
#     ---- narrow the sacle of limits and stop
#     :param price:
#     :param prefer_round_down:
#     :param tick_size:
#     :param diff:
#     :return:
#     """
#     # return 小数位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)
#     # 保留tick_size
#     diff *= multiplier
#     # 保留系统精度
#     epsilon = sys.float_info * 10
#     diff = diff - epsilon
#
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded


# def order(self, asset, amount, style =None, order_id=None):
#     """Place an order.
#
#     Parameters
#     ----------
#     asset : zipline.asset.Asset
#         The asset that this order is for.
#     amount : int
#         The amount of shares to order. If ``amount`` is positive, this is
#         the number of shares to buy or cover. If ``amount`` is negative,
#         this is the number of shares to sell or short.
#     style : zipline.finance.execution.ExecutionStyle
#         The execution style for the order.
#     order_id : str, optional
#         The unique identifier for this order.
#
#     Returns
#     -------
#     order_id : str or None
#         The unique identifier for this order, or None if no order was
#         placed.
#
#     Notes
#     -----
#     amount > 0 :: Buy/Cover
#     amount < 0 :: Sell/Short
#     Market order:    order(asset, amount)
#     Limit order:     order(asset, amount, style=LimitOrder(limit_price))
#     Stop order:      order(asset, amount, style=StopOrder(stop_price))
#     StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,
#                            stop_price))
#     """
#     # something could be done with amount to further divide
#     # between buy by share count OR buy shares up to a dollar amount
#     # numeric == share count  AND  "$dollar.cents" == cost amount
#
#     if amount == 0:
#         # Don't bother placing orders for 0 shares.
#         return None
#     elif amount > self.max_shares:
#         # Arbitrary limit of 100 billion (US) shares will never be
#         # exceeded except by a buggy algorithm.
#         raise OverflowError("Can't order more than %d shares" %
#                             self.max_shares)
#
#     is_buy = (amount > 0)
#     order = Order(
#         dt=self.current_dt,
#         asset=asset,
#         amount=amount,
#         stop=style.get_stop_price(is_buy),
#         limit=style.get_limit_price(is_buy),
#         id=order_id
#     )
#
#     self.open_orders[order.asset].append(order)
#     self.orders[order.id] = order


# full_share_count = self.amount * float(ratio)
# new_cost_basics = round(self.cost_basis / float(ratio), 2)
# left_cash = (full_share_count - np.floor(full_share_count)) * new_cost_basics
# self.cost_basis = np.floor(new_cost_basics)
# self.amount = full_share_count
# return left_cash

# def update(self,txn):
#     if self.asset != txn.asset:
#         raise Exception('transaction must be the same with position asset')
#
#     if self.last_sale_dt is None or txn.dt > self.last_sale_dt:
#         self.last_sale_dt = txn.dt
#         self.last_sale_price = txn.price
#
#     total_shares = txn.amount + self.amount
#     if total_shares == 0:
#         # 用于统计transaction是否盈利
#         # self.cost_basis = 0.0
#         position_return = (txn.price - self.cost_basis)/self.cost_basis
#         self.cost_basis = position_return
#     elif total_shares < 0:
#         raise Exception('for present put action is not allowed')
#     else:
#         total_cost = txn.amout * txn.price + self.amount * self.cost_basis
#         new_cost_basis = total_cost / total_shares
#         self.cost_basis = new_cost_basis
#
#     self.amount = total_shares

# def update_position(self,
#                     asset,
#                     amount = None,
#                     last_sale_price = None,
#                     last_sale_date = None,
#                     cost_basis = None):
#     self._dirty_stats = True
#
#     try:
#         position = self.positions[asset]
#     except KeyError:
#         position = Position(asset)
#
#     if amount is not None:
#         position.amount = amount
#     if last_sale_price is not None :
#         position.last_sale_price = last_sale_price
#     if last_sale_date is not None :
#         position.last_sale_date = last_sale_date
#     if cost_basis is not None :
#         position.cost_basis = cost_basis
#
# # 执行
# def execute_transaction(self,txn):
#     self._dirty_stats = True
#
#     asset = txn.asset
#
#     # 新的股票仓位
#     if asset not in self.positions:
#         position = Position(asset)
#     else:
#         position = self.positions[asset]
#
#     position.update(txn)
#
#     if position.amount ==0 :
#         #统计策略的对应的收益率
#         dt = txn.dt
#         algorithm_ret = position.cost_basis
#         asset_origin = position.asset.reason
#         self.record_vars[asset_origin] = {str(dt):algorithm_ret}
#
#         del self.positions[asset]

# def handle_spilts(self,splits):
#     total_leftover_cash = 0
#
#     for asset,ratio in splits.items():
#         if asset in self.positions:
#             position = self.positions[asset]
#             leftover_cash = position.handle_split(asset,ratio)
#             total_leftover_cash += leftover_cash
#     return total_leftover_cash

# 将分红或者配股的数据分类存储
# def earn_divdends(self,cash_divdends,stock_divdends):
#     """
#         given a list of divdends where ex_date all the next_trading
#         including divdend and stock_divdend
#     """
#     for cash_divdend in cash_divdends:
#         div_owned = self.positions[cash_divdend['paymen_asset']].earn_divdend(cash_divdend)
#         self._unpaid_divdend[cash_divdend.pay_date].apppend(div_owned)
#
#     for stock_divdend in stock_divdends:
#         div_owned_ = self.positions[stock_divdend['payment_asset']].earn_stock_divdend(stock_divdend)
#         self._unpaid_stock_divdends[stock_divdend.pay_date].append(div_owned_)

# 根据时间执行分红或者配股
# def pay_divdends(self,next_trading_day):
#     """
#         股权登记日，股权除息日（为股权登记日下一个交易日）
#         但是红股的到账时间不一致（制度是固定的）
#         根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
#         投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
#         其中对于分红的时间存在差异
#
#         根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，
#
#         持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
#     """
#     net_cash_payment = 0.0
#
#     # cash divdend
#     try:
#         payments = self._unpaid_divdend[next_trading_day]
#         del self._unpaid_divdend[next_trading_day]
#     except KeyError:
#         payments = []
#
#     for payment in payments:
#         net_cash_payment += payment['cash_amount']
#
#     #stock divdend
#     try:
#         stock_payments = self._unpaid_stock_divdends[next_trading_day]
#     except KeyError:
#         stock_payments = []
#
#     for stock_payment in stock_payments:
#         payment_asset = stock_payment['payment_asset']
#         share_amount = stock_payment['share_amount']
#         if payment_asset in self.positions:
#             position = self.positions[payment_asset]
#         else:
#             position = self.positions[payment_asset] = Position(payment_asset)
#         position.amount  += share_amount
#     return net_cash_payment

# def calculate_position_tracker_stats(positions,stats):
#     """
#         stats ---- PositionStats
#     """
#     longs_count = 0
#     long_exposure = 0
#     shorts_count = 0
#     short_exposure = 0
#
#     for outer_position in positions.values():
#         position = outer_position.inner_position
#         #daily更新价格
#         exposure = position.amount * position.last_sale_price
#         if exposure > 0:
#             longs_count += 1
#             long_exposure += exposure
#         elif exposure < 0:
#             shorts_count +=1
#             short_exposure += exposure
#     #
#     net_exposure = long_exposure + short_exposure
#     gross_exposure = long_exposure - short_exposure
#
#     stats.gross_exposure = gross_exposure
#     stats.long_exposure = long_exposure
#     stats.longs_count = longs_count
#     stats.net_exposure = net_exposure
#     stats.short_exposure = short_exposure
#     stats.shorts_count = shorts_count

# def process_transaction(self,transaction):
#     position = self.position_tracker.positions[asset]
#     amount = position.amount
#     left_amount = amount + transaction.amount
#     if left_amount == 0:
#         self._cash_flow( - self.commission.calculate(transaction))
#         del self._payout_last_sale_price[asset]
#     elif left_amount < 0:
#         raise Exception('禁止融券卖出')
#     # calculate cash
#     self._cash_flow( - transaction.amount * transaction.price)
#     #execute transaction
#     self.position_tracker.execute_transaction(transaction)
#     transaction_dict = transaction.to_dict()
#     self._processed_transaction[transaction.dt].append(transaction_dict)

# def process_commission(self,commission):
#     asset = commission['asset']
#     cost = commission['cost']
#
#     self.position_tracker.handle_commission(asset,cost)
#     self._cash_flow(-cost)

# def process_split(self,splits):
#     """
#         splits --- (asset,ratio)
#     :param splits:
#     :return:
#     """
#     leftover_cash = self.position_tracker.handle_spilts(splits)
#     if leftover_cash > 0 :
#         self._cash_flow(leftover_cash)
#
# def process_divdends(self,next_session,adjustment_reader):
#     """
#         基于时间、仓位获取对应的现金分红、股票分红
#     """
#     position_tracker = self.position_tracker
#     #针对字典 --- set return keys
#     held_sids = set(position_tracker.positions)
#     if held_sids:
#         cash_divdend = adjustment_reader.get_dividends_with_ex_date(
#             held_sids,
#             next_session,
#         )
#         stock_dividends = (
#             adjustment_reader.get_stock_dividends_with_ex_date(
#                 held_sids,
#                 next_session,
#             )
#         )
#     #添加
#     position_tracker.earn_divdends(
#         cash_divdend,stock_dividends
#     )
#     #基于session --- pay_date 处理
#     self._cash_flow(
#         position_tracker.pay_divdends(next_session)
#     )
# self.record_vars
# def update_portfolio(self):
#     """
#         force a computation of the current portfolio
#         portofolio 保留最新
#     """
#     if not self._dirty_portfolio:
#         return
#
#     portfolio = self._portfolio
#     pt = self.position_tracker
#
#     portfolio.positions = pt.get_positions()
#     #计算positioin stats --- sync_last_sale_price
#     position_stats = pt.stats
#
#     portfolio.positions_value = position_value = (
#         position_stats.net_value
#     )
#
#     portfolio.positions_exposure = position_stats.net_exposure
#     self._cash_flow(self._get_payout_total(pt.positions))
#
#     # portfolio_value 初始化capital_value
#     start_value = portfolio.portfolio_value
#     portfolio.portfolio_value = end_value = portfolio.cash + position_value
#
#     # daily 每天账户净值波动
#     pnl = end_value - start_value
#     if start_value !=0 :
#         returns = pnl/start_value
#     else:
#         returns = 0.0
#
#     #pnl --- 投资收益
#     portfolio.pnl += pnl
#     # 每天的区间收益率 --- 递归方式
#     portfolio.returns = (
#         (1+portfolio.returns) *
#         (1+returns) - 1
#     )
#     self._dirty_portfolio = False
# for asset, old_price in payout_last_sale_prices.items():
#     position = positions[asset]
#     payout_last_sale_prices[asset] = price = position.last_sale_price
#     amount = position.amount
#     total += calculate_payout(
#         amount,
#         old_price,
#         price,
#         asset.price_multiplier,
#     )
# return total
import numpy as np
from scipy.optimize import fsolve
#
# a = [15.3,14.7,14.9,14.01,15.2,16.7,16.9]
# print(np.std(a))
# b = np.std(a)
#
#
# def func(paramlist):
#
#     a,b =paramlist[0],paramlist[1]
#     return [a / (a+b) - 0.0476,
#             (a*b) /((a+b+1) * (a+b) ** 2) - 0.0021]
# c1,c2=fsolve(func,[0,0])
# print(c1,c2)
# e = c1 / (c1+c2)


# a = 10
# b = 200
# e = a/(a + b)
# s = (a*b) /((a+b+1) * (a+b) ** 2)
# print(e,s)
# pct = [0.02,-0.03,0.04,0.05,0.08,-0.06,0.07]
# print(np.std(pct))
# def override_account_fields(self,
#                             settled_cash=not_overridden,
#                             total_positions_values=not_overridden,
#                             total_position_exposure=not_overridden,
#                             cushion=not_overridden,
#                             gross_leverage=not_overridden,
#                             net_leverage=not_overridden,
#                             ):
#     # locals ---函数内部的参数
#     self._account_overrides = kwargs = {k: v for k, v in locals().items() if v is not not_overridden}
#     del kwargs['self']
# for k, v in self._account_overrides:
#     setattr(account, k, v)
from itertools import product

# p = {'a':[1,2,3],'b':[4,5,6],'c':[7,8,9]}
# items = p.items()
#
# keys, values = zip(*items)
# print(keys)
# print(values)
# #product --- 每个列表里面取一个元素
# for v in product(*values):
#     params = dict(zip(keys, v))
#     print(params)
#
# from toolz import concatv
#
# list(concatv([], ["a"], ["b", "c"]))
# #['a', 'b', 'c']
#
# #代码高亮
# try:
#     from pygments import highlight
#     from pygments.lexers import PythonLexer
#     from pygments.formatters import TerminalFormatter
#     PYGMENTS = True
# except ImportError:
#     PYGMENTS = False
#

#
# class NoHooks(PipelineHooks):
#     """A PipelineHooks that defines no-op methods for all available hooks.
#     """
#     @contextmanager
#     def running_pipeline(self, pipe, start_date, end_date):
#         yield
#
#     @contextmanager
#     def computing_chunk(self, terms, start_date, end_date):
#         yield
#
#     @contextmanager
#     def loading_terms(self, terms):
#         yield
#
#     @contextmanager
#     def computing_term(self, term):
#         yield

# @contextmanager
# @abstractmethod
# def loading_terms(self, terms):
#     """Contextmanager entered when loading a batch of LoadableTerms.
#
#     Parameters
#     ----------
#     terms : list[zipline.pipe.LoadableTerm]
#         Terms being loaded.
#     """
#
# @contextmanager
# @abstractmethod
# def computing_term(self, term):
#     """Contextmanager entered when computing a ComputableTerm.
#
#     Parameters
#     ----------
#     terms : zipline.pipe.ComputableTerm
#         Terms being computed.
#     """
#
# def delegating_hooks_method(method_name):
#     """Factory function for making DelegatingHooks methods.
#     """
#     if method_name in PIPELINE_HOOKS_CONTEXT_MANAGERS:
#         # Generate a contextmanager that enters the context of all child hooks.
#         # wraps --- callable
#         @wraps(getattr(PipelineHooks, method_name))
#         @contextmanager
#         def ctx(self, *args, **kwargs):
#             with ExitStack() as stack:
#                 for hook in self._hooks:
#                     sub_ctx = getattr(hook, method_name)(*args, **kwargs)
#                     stack.enter_context(sub_ctx)
#                 yield stack
#         return ctx
#     else:
#         # Generate a method that calls methods of all child hooks.
#         @wraps(getattr(PipelineHooks, method_name))
#         def method(self, *args, **kwargs):
#             for hook in self._hooks:
#                 sub_method = getattr(hook, method_name)
#                 sub_method(*args, **kwargs)
#
#         return method
#
#
# class DelegatingHooks(PipelineHooks):
#     """A PipelineHooks that delegates to one or more other hooks.
#
#     Parameters
#     ----------
#     hooks : list[implements(PipelineHooks)]
#         Sequence of hooks to delegate to.
#     """
#     def __new__(cls, hooks):
#         if len(hooks) == 0:
#             # OPTIMIZATION: Short-circuit to a NoHooks if we don't have any
#             # sub-hooks.
#             return NoHooks()
#         else:
#             self = super(DelegatingHooks, cls).__new__(cls)
#             self._hooks = hooks
#             return self
#
#     # Implement all interface methods by delegating to corresponding methods on
#     # input hooks. locals --- __dict__ 覆盖原来的方法
#     locals().update({
#         name: delegating_hooks_method(name)
#         # TODO: Expose this publicly on interface.
#         for name in PipelineHooks._signatures
#     })
#
#
# del delegating_hooks_method
#
# class AlgorithmSimulator(object):
#
#     EMISSION_TO_PERF_KEY_MAP = {
#         'minute': 'minute_perf',
#         'daily': 'daily_perf'
#     }
#
#     def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
#                  restrictions, universe_func):
#
#         # ==============
#         # ArkQuant
#         # Param Setup
#         # ==============
#         self.sim_params = sim_params
#         self.data_portal = data_portal
#         self.restrictions = restrictions
#
#         # ==============
#         # Algo Setup
#         # ==============
#         self.algo = algo
#
#         # ==============
#         # Snapshot Setup
#         # ==============
#
#         # We don't have a datetime for the current snapshot until we
#         # receive a message.
#         self.simulation_dt = None
#
#         self.clock = clock
#
#         self.benchmark_source = benchmark_source
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#         # This object is the way that user algorithms interact with OHLCV data,
#         # fetcher data, and some API methods like `data.can_trade`.
#         self.current_data = self._create_bar_data(universe_func)
#
#     def get_simulation_dt(self):
#         return self.simulation_dt
#
#     #获取日数据，封装为一个API(fetch process flush other api)
#     def _create_bar_data(self, universe_func):
#         return BarData(
#             data_portal=self.data_portal,
#             simulation_dt_func=self.get_simulation_dt,
#             data_frequency=self.sim_params.data_frequency,
#             trading_calendar=self.algo.trading_calendar,
#             restrictions=self.restrictions,
#             universe_func=universe_func
#         )
#
#     def transform(self):
#         """
#         Main generator work loop.
#         """
#         algo = self.algo
#         metrics_tracker = algo.metrics_tracker
#         emission_rate = metrics_tracker.emission_rate
#
#         #生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
#         def every_bar(dt_to_use, current_data=self.current_data,
#                       handle_data=algo.event_manager.handle_data):
#             for capital_change in calculate_minute_capital_changes(dt_to_use):
#                 yield capital_change
#
#             self.simulation_dt = dt_to_use
#             # called every tick (minute or day).
#             algo.on_dt_changed(dt_to_use)
#
#             broker = algo.broker
#
#             # handle any transactions and commissions coming out new orders
#             # placed in the last bar
#             new_transactions, new_commissions, closed_orders = \
#                 broker.get_transactions(current_data)
#
#             broker.prune_orders(closed_orders)
#
#             for transaction in new_transactions:
#                 metrics_tracker.process_transaction(transaction)
#
#                 # since this order was modified, record it
#                 order = broker.orders[transaction.order_id]
#                 metrics_tracker.process_order(order)
#
#             for commission in new_commissions:
#                 metrics_tracker.process_commission(commission)
#
#             handle_data(algo, current_data, dt_to_use)
#
#             # grab any new orders from the broker, then clear the list.
#             # this includes cancelled orders.
#             new_orders = broker.new_orders
#             broker.new_orders = []
#
#             # if we have any new orders, record them so that we know
#             # in what perf period they were placed.
#             for new_order in new_orders:
#                 metrics_tracker.process_order(new_order)
#
#         def once_a_day(midnight_dt, current_data=self.current_data,
#                        data_portal=self.data_portal):
#             # process any capital changes that came overnight
#             for capital_change in algo.calculate_capital_changes(
#                     midnight_dt, emission_rate=emission_rate,
#                     is_interday=True):
#                 yield capital_change
#
#             # set all the timestamps
#             self.simulation_dt = midnight_dt
#             algo.on_dt_changed(midnight_dt)
#
#             metrics_tracker.handle_market_open(
#                 midnight_dt,
#                 algo.data_portal,
#             )
#
#             # handle any splits that impact any positions or any open orders.
#             assets_we_care_about = (
#                 viewkeys(metrics_tracker.positions) |
#                 viewkeys(algo.broker.open_orders)
#             )
#
#             if assets_we_care_about:
#                 splits = data_portal.get_splits(assets_we_care_about,
#                                                 midnight_dt)
#                 if splits:
#                     algo.broker.process_splits(splits)
#                     metrics_tracker.handle_splits(splits)
#
#         def on_exit():
#             # Remove references to algo, data portal, et al to break cycles
#             # and ensure deterministic cleanup of these objects when the
#             # ArkQuant finishes.
#             self.algo = None
#             self.benchmark_source = self.current_data = self.data_portal = None
#
#         with ExitStack() as stack:
#             """
#             由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
#             这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
#             enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
#             callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
#             """
#             stack.callback(on_exit)
#             stack.enter_context(self.processor)
#             stack.enter_context(ZiplineAPI(self.algo))
#
#             if algo.data_frequency == 'minute':
#                 def execute_order_cancellation_policy():
#                     algo.broker.execute_cancel_policy(SESSION_END)
#
#                 def calculate_minute_capital_changes(dt):
#                     # process any capital changes that came between the last
#                     # and current minutes
#                     return algo.calculate_capital_changes(
#                         dt, emission_rate=emission_rate, is_interday=False)
#             else:
#                 def execute_order_cancellation_policy():
#                     pass
#
#                 def calculate_minute_capital_changes(dt):
#                     return []
#
#             for dt, action in self.clock:
#                 if action == BAR:
#                     for capital_change_packet in every_bar(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_START:
#                     for capital_change_packet in once_a_day(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_END:
#                     # End of the session.
#                     positions = metrics_tracker.positions
#                     position_assets = algo.asset_finder.retrieve_all(positions)
#                     self._cleanup_expired_assets(dt, position_assets)
#
#                     execute_order_cancellation_policy()
#                     algo.validate_account_controls()
#
#                     yield self._get_daily_message(dt, algo, metrics_tracker)
#                 elif action == BEFORE_TRADING_START_BAR:
#                     self.simulation_dt = dt
#                     algo.on_dt_changed(dt)
#                     algo.before_trading_start(self.current_data)
#                 elif action == MINUTE_END:
#                     minute_msg = self._get_minute_message(
#                         dt,
#                         algo,
#                         metrics_tracker,
#                     )
#
#                     yield minute_msg
#
#             risk_message = metrics_tracker.handle_simulation_end(
#                 self.data_portal,
#             )
#             yield risk_message
#
#     def _cleanup_expired_assets(self, dt, position_assets):
#         """
#         Clear out any asset that have expired before starting a new sim day.
#
#         Performs two functions:
#
#         1. Finds all asset for which we have open orders and clears any
#            orders whose asset are on or after their auto_close_date.
#
#         2. Finds all asset for which we have positions and generates
#            close_position events for any asset that have reached their
#            auto_close_date.
#         """
#         algo = self.algo
#
#         def past_auto_close_date(asset):
#             acd = asset.auto_close_date
#             return acd is not None and acd <= dt
#
#         # Remove positions in any sids that have reached their auto_close date.
#         assets_to_clear = \
#             [asset for asset in position_assets if past_auto_close_date(asset)]
#         metrics_tracker = algo.metrics_tracker
#         data_portal = self.data_portal
#         for asset in assets_to_clear:
#             metrics_tracker.process_close_position(asset, dt, data_portal)
#
#         # Remove open orders for any sids that have reached their auto close
#         # date. These orders get processed immediately because otherwise they
#         # would not be processed until the first bar of the next day.
#         broker = algo.broker
#         assets_to_cancel = [
#             asset for asset in broker.open_orders
#             if past_auto_close_date(asset)
#         ]
#         for asset in assets_to_cancel:
#             broker.cancel_all_orders_for_asset(asset)
#
#         # Make a copy here so that we are not modifying the list that is being
#         # iterated over.
#         for order in copy(broker.new_orders):
#             if order.status == ORDER_STATUS.CANCELLED:
#                 metrics_tracker.process_order(order)
#                 broker.new_orders.remove(order)
#
#     def _get_daily_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         perf_message = metrics_tracker.handle_market_close(
#             dt,
#             self.data_portal,
#         )
#         perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
#         return perf_message
#
#     def _get_minute_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         rvars = algo.recorded_vars
#
#         minute_message = metrics_tracker.handle_minute_close(
#             dt,
#             self.data_portal,
#         )
#
#         minute_message['minute_perf']['recorded_vars'] = rvars
#         return minute_message
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#
# class PeriodLabel(object):
#     """Backwards compat, please kill me.
#     """
#     def start_of_session(self, ledger, session, data_portal):
#         self._label = session.strftime('%Y-%m')
#
#     def end_of_bar(self, packet, *args):
#         packet['cumulative_risk_metrics']['period_label'] = self._label
#
#     end_of_session = end_of_bar
#
#
# class _ConstantCumulativeRiskMetric(object):
#     """A metric which does not change, ever.
#
#     Notes
#     -----
#     This exists to maintain the existing structure of the perf packets. We
#     should kill this as soon as possible.
#     """
#     def __init__(self, field, value):
#         self._field = field
#         self._value = value
#
#     def start_of_session(self, packet,*args):
#         packet['cumulative_risk_metrics'][self._field] = self._value
#
#     def end_of_session(self, packet, *args):
#         packet['cumulative_risk_metrics'][self._field] = self._value


# If you are adding new attributes, don't update this set. This method
# is deprecated to normal attribute access so we don't want to encourage
# new usages.
# __getitem__ = _deprecated_getitem_method(
#     'portfolio', {
#         'capital_used',
#         'starting_cash',
#         'portfolio_value',
#         'pnl',
#         'returns',
#         'cash',
#         'positions',
#         'start_date',
#         'positions_value',
#     },
# )

#toolz.itertoolz.groupby(key, seq)
from dateutil.relativedelta import relativedelta
# import datetime , pandas as pd
#
# start_session = datetime.datetime.strptime('2010-01-31','%Y-%m-%d')
# end_session = datetime.datetime.strptime('2012-01-31','%Y-%m-%d')
#
# print(start_session,end_session)
#
# # end = end_session.replace(day=1) + relativedelta(months=1)
# end = end_session
# print(end)
#
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end,
#     freq='M',
#     tz='utc',
#     closed = 'left'
# )
# print('months',months.size)
# print(type(months),months)
# months.iloc[-1] = 'c_test'
# period = months[0].to_period(freq='%dM' % 3)
# print(months[::3])
# print('period',period.end_date)


# for period_timestamp in months:
#     period = period_timestamp.to_period(freq='%dM' % months_per)

# # 下个月第一天
# end = end_session.replace(day=1) + relativedelta(months=1)
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end - datetime.timedelta(days=1),
#     freq='M',
#     tz='utc',
# )

# from sys import float_info
#
# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#     Asymmetric rounding function for adjusting prices to the specified number
#     of places in a way that "improves" the price. For limit prices, this means
#     preferring to round down on buys and preferring to round up on sells.
#     For stop prices, it means the reverse.
#
#     If prefer_round_down == True:
#         When .05 below to .95 above a specified decimal place, use it.
#     If prefer_round_down == False:
#         When .95 below to .05 above a specified decimal place, use it.
#
#     In math-speak:
#     If prefer_round_down: [<X-1>.0095, X.0195) -> round to X.01.
#     If not prefer_round_down: (<X-1>.0005, X.0105] -> round to X.01.
#     """
#     # 返回位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)  # adjust diff to precision of tick size
#     diff *= multiplier  # adjust diff to value of tick_size
#
#     # Subtracting an epsilon from diff to enforce the open-ness of the upper
#     # bound on buys and the lower bound on sells.  Using the actual system
#     # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
#     epsilon = float_info.epsilon * 10
#     diff = diff - epsilon
#
#     # relies on rounding half away from zero, unlike numpy's bankers' rounding
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded
#
#
# # 生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
# def every_bar(dt_to_use, current_data=self.current_data,
#               handle_data=algo.event_manager.handle_data):
#     for capital_change in calculate_minute_capital_changes(dt_to_use):
#         yield capital_change
#
#     self.simulation_dt = dt_to_use
#     # called every tick (minute or day).
#     algo.on_dt_changed(dt_to_use)
#
#     broker = algo.broker
#
#     # handle any transactions and commissions coming out new orders
#     # placed in the last bar
#     new_transactions, new_commissions, closed_orders = \
#         broker.get_transactions(current_data)
#
#     broker.prune_orders(closed_orders)
#
#     for transaction in new_transactions:
#         metrics_tracker.process_transaction(transaction)
#
#         # since this order was modified, record it
#         order = broker.orders[transaction.order_id]
#         metrics_tracker.process_order(order)
#
#     for commission in new_commissions:
#         metrics_tracker.process_commission(commission)
#
#     handle_data(algo, current_data, dt_to_use)
#
#     # grab any new orders from the broker, then clear the list.
#     # this includes cancelled orders.
#     new_orders = broker.new_orders
#     broker.new_orders = []
#
#     # if we have any new orders, record them so that we know
#     # in what perf period they were placed.
#     for new_order in new_orders:
#         metrics_tracker.process_order(new_order)
#
# def once_a_day(midnight_dt, current_data=self.current_data,
#                data_portal=self.data_portal):
#     # process any capital changes that came overnight
#     for capital_change in algo.calculate_capital_changes(
#             midnight_dt, emission_rate=emission_rate,
#             is_interday=True):
#         yield capital_change
#
#     # set all the timestamps
#     self.simulation_dt = midnight_dt
#     algo.on_dt_changed(midnight_dt)
#
#     metrics_tracker.handle_market_open(
#         midnight_dt,
#         algo.data_portal,
#     )
#
#     # handle any splits that impact any positions or any open orders.
#     assets_we_care_about = (
#         viewkeys(metrics_tracker.positions) |
#         viewkeys(algo.broker.open_orders)
#     )
#
#     if assets_we_care_about:
#         splits = data_portal.get_splits(assets_we_care_about,
#                                         midnight_dt)
#         if splits:
#             algo.broker.process_splits(splits)
#             metrics_tracker.handle_splits(splits)
#

# -*- coding:utf-8 -*-

# import unittest
#
# class NamesTestCase(unittest.TestCase):
#     """
#         所有以test_开头的方法都会自动运行
#         assertEqual,assertNotEqual,assertTrue,assertFalse,assertIn,assertNotIn
#         setUp -- called before c_test method ; setUpClass --A  class method called before tests in an individual class are run
#     """
#     @classmethod
#     def setUpClass(cls) -> None:
#         pass
#
#     def test_first_last_name(self):
#         pass
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         pass


# @property
# def birth(self):
#     return self._birth
#
# @birth.setter
# def birth(self, value):
#     self._birth = value
#
# @birth.getter
# def birth(self):
#     return self._birth
#
# getter  ---- property ;  setter --- @func.setter
#
# __delete__(instance), __get__(instance,owner) , __set__(instance,value) 描述器 , 实例为类的类属性
# __getattribute__ --- __getattr__ (显式访问不存在饿属性,除非显示调用或引发AttributeError异常） ）
#
#
# __delete__(self,instance) ,__del__(self)
#
# math.copysign(x, y)
# Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0.


# Get the first trading minute
# self._first_trading_minute, _ = (
#     _calendar.open_and_close_for_session(
#         [self._first_trading_day]
#     )
#     if self._first_trading_day is not None else (None, None)
# )
#
# # Store the locs of the first day and first minute
# self._first_trading_day_loc = (
#     _calendar.all_sessions.get_loc(self._first_trading_day)
#     if self._first_trading_day is not None else None
# )
# -*- coding : utf-8 -*-
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# index=True,
# index_label=first(tbl.primary_key.columns).name,

# if self.symbol:
#     return '%s(%d [%s])' % (type(self).__name__, self.sid, self.symbol)
# else:
#     return '%s(%d)' % (type(self).__name__, self.sid)

# def _compute_date_range_slice(self, start_date, end_date):
#     # Get the index of the start of dates for ``start_date``.
#     start_ix = self.dates.searchsorted(start_date)
#
#     # Get the index of the start of the first date **after** end_date.
#     end_ix = self.dates.searchsorted(end_date, side='right')
#
#     return slice(start_ix, end_ix)
# ins = ins.groupby(table.c.sid)


# @api_method
# def fetch_csv(self,
#               url,
#               pre_func=None,
#               post_func=None,
#               date_column='date',
#               date_format=None,
#               timezone=pytz.utc.zone,
#               symbol=None,
#               mask=True,
#               symbol_column=None,
#               special_params_checker=None,
#               country_code=None,
#               **kwargs):
#     """Fetch a csv from a remote url and register the data so that it is
#     queryable from the ``data`` object.
#
#     Parameters
#     ----------
#     url : str
#         The url of the csv file to load.
#     pre_func : callable[pd.DataFrame -> pd.DataFrame], optional
#         A callback to allow preprocessing the raw data returned from
#         fetch_csv before dates are paresed or symbols are mapped.
#     post_func : callable[pd.DataFrame -> pd.DataFrame], optional
#         A callback to allow postprocessing of the data after dates and
#         symbols have been mapped.
#     date_column : str, optional
#         The name of the column in the preprocessed dataframe containing
#         datetime information to map the data.
#     date_format : str, optional
#         The format of the dates in the ``date_column``. If not provided
#         ``fetch_csv`` will attempt to infer the format. For information
#         about the format of this string, see :func:`pandas.read_csv`.
#     timezone : tzinfo or str, optional
#         The timezone for the datetime in the ``date_column``.
#     symbol : str, optional
#         If the data is about a new asset or index then this string will
#         be the name used to identify the values in ``data``. For example,
#         one may use ``fetch_csv`` to load data for VIX, then this field
#         could be the string ``'VIX'``.
#     mask : bool, optional
#         Drop any rows which cannot be symbol mapped.
#     symbol_column : str
#         If the data is attaching some new attribute to each asset then this
#         argument is the name of the column in the preprocessed dataframe
#         containing the symbols. This will be used along with the date
#         information to map the sids in the asset finder.
#     country_code : str, optional
#         Country code to use to disambiguate symbol lookups.
#     **kwargs
#         Forwarded to :func:`pandas.read_csv`.
#
#     Returns
#     -------
#     csv_data_source : zipline.sources.requests_csv.PandasRequestsCSV
#         A requests source that will pull data from the url specified.
#     """
#     if country_code is None:
#         country_code = self.default_fetch_csv_country_code(
#             self.trading_calendar,
#         )
#
#     # Show all the logs every time fetcher is used.
#     csv_data_source = PandasRequestsCSV(
#         url,
#         pre_func,
#         post_func,
#         self.asset_finder,
#         self.trading_calendar.day,
#         self.sim_params.start_session,
#         self.sim_params.end_session,
#         date_column,
#         date_format,
#         timezone,
#         symbol,
#         mask,
#         symbol_column,
#         data_frequency=self.data_frequency,
#         country_code=country_code,
#         special_params_checker=special_params_checker,
#         **kwargs
#     )
#
#     # ingest this into dataportal
#     self.data_portal.handle_extra_source(csv_data_source.df,
#                                          self.sim_params)
#
#     return csv_data_source

#
# print(__file__)

# ####################
# # Account Controls #
# ####################
#
# def register_account_control(self, control):
#     """
#     Register a new AccountControl to be checked on each bar.
#     """
#     if self.initialized:
#         raise RegisterAccountControlPostInit()
#     self.account_controls.append(control)
#
# def validate_account_controls(self):
#     for control in self.account_controls:
#         control.validate(self.portfolio,
#                          self.account,
#                          self.get_datetime(),
#                          self.trading_client.current_data)
#

# return callable fetch attr from its operant
# operator.attrgetter(return a callable that fetches attr from operant)
# operate.itemgetter (return a callable that uses method __getitem__())
# rsplit 从右往左 参数 sep(默认为所有空字符) count=count（sep）

# # 根据dt获取change,动态计算，更新数据
# def calculate_capital_changes(self, dt, emission_rate, is_interday,
#                               portfolio_value_adjustment=0.0):
#     """
#     If there is a capital change for a given dt, this means the the change
#     occurs before `handle_data` on the given dt. In the case of the
#     change being a target value, the change will be computed on the
#     portfolio value according to prices at the given dt
#
#     `portfolio_value_adjustment`, if specified, will be removed from the
#     portfolio_value of the cumulative performance when calculating deltas
#     from target capital changes.
#     """
#     try:
#         capital_change = self.capital_changes[dt]
#     except KeyError:
#         return
#
#     self._sync_last_sale_prices()
#     if capital_change['type'] == 'target':
#         target = capital_change['value']
#         capital_change_amount = (
#             target -
#             (
#                 self.portfolio.portfolio_value -
#                 portfolio_value_adjustment
#             )
#         )
#
#         logging.log.info('Processing capital change to target %s at %s. Capital '
#                  'change delta is %s' % (target, dt,
#                                          capital_change_amount))
#     elif capital_change['type'] == 'delta':
#         target = None
#         capital_change_amount = capital_change['value']
#         logging.log.info('Processing capital change of delta %s at %s'
#                  % (capital_change_amount, dt))
#     else:
#         logging.log.error("Capital change %s does not indicate a valid type "
#                   "('target' or 'delta')" % capital_change)
#         return
#
#     self.capital_change_deltas.update({dt: capital_change_amount})
#     self.metrics_tracker.capital_change(capital_change_amount)
#
#     yield {
#         'capital_change':
#             {'date': dt,
#              'type': 'cash',
#              'target': target,
#              'delta': capital_change_amount}
#     }
from copy import copy
from datetime import tzinfo
import logging

# @api_method
# @preprocess(tz=coerce_string(pytz.timezone))
# @expect_types(tz=optional(tzinfo))
# def get_datetime(self, tz=None):
#     """
#     Returns the current ArkQuant datetime.
#
#     Parameters
#     ----------
#     tz : tzinfo or str, optional
#         The timezone to return the datetime in. This defaults to utc.
#
#     Returns
#     -------
#     dt : datetime
#         The current ArkQuant datetime converted to ``tz``.
#     """
#     dt = self.datetime
#     assert dt.tzinfo == pytz.utc, "algorithm should have a utc datetime"
#     if tz is not None:
#         dt = dt.astimezone(tz)
#     return dt

# control = RestrictedListOrder(on_error, restrictions)
# self.register_trading_control(control)
# self.restrictions |= restrictions

# sa.ForeignKey(equity_basics.c.sid),

# grouped_by_sid = source_df.groupby(["sid"])
# group_names = grouped_by_sid.groups.keys()
# group_dict = {}
# for group_name in group_names:
#     group_dict[group_name] = grouped_by_sid.get_group(group_name)
# for col_name in df.columns.difference(['sid']):

# """
# Construction of sentinel objects.
#
# Sentinel objects are used when you only care to check for object identity.
# """
# import sys
# from textwrap import dedent
#
#
# class _Sentinel(object):
#     """Base class for Sentinel objects.
#     """
#     __slots__ = ('__weakref__',)
#
#
# def is_sentinel(obj):
#     return isinstance(obj, _Sentinel)
#
#
# # 返回 目标的具体信息文件名、行号基于_getframe
# def sentinel(name, doc=None):
#     try:
#         value = sentinel._cache[name]  # memoized
#     except KeyError:
#         pass
#     else:
#         if doc == value.__doc__:
#             return value
#
#         raise ValueError(dedent(
#             """\
#             New sentinel value %r conflicts with an existing sentinel of the
#             same name.
#             Old sentinel docstring: %r
#             New sentinel docstring: %r
#
#             The old sentinel was created at: %s
#
#             Resolve this conflict by changing the name of one of the sentinels.
#             """,
#         ) % (name, value.__doc__, doc, value._created_at))
#
#     try:
#         frame = sys._getframe(1)
#     except ValueError:
#         frame = None
#
#     if frame is None:
#         created_at = '<unknown>'
#     else:
#         created_at = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)
#
#     @object.__new__   # bind a single instance to the name 'Sentinel'
#     class Sentinel(_Sentinel):
#         __doc__ = doc
#         __name__ = name
#
#         # store created_at so that we can report this in case of a duplicate
#         # name violation
#         _created_at = created_at
#
#         def __new__(cls):
#             raise TypeError('cannot create %r instances' % name)
#
#         def __repr__(self):
#             return 'sentinel(%r)' % name
#
#         def __reduce__(self):
#             return sentinel, (name, doc)
#
#         def __deepcopy__(self, _memo):
#             return self
#
#         def __copy__(self):
#             return self
#
#     cls = type(Sentinel)
#     try:
#         cls.__module__ = frame.f_globals['__name__']
#     except (AttributeError, KeyError):
#         # Couldn't get the name from the calling scope, just use None.
#         # AttributeError is when frame is None, KeyError is when f_globals
#         # doesn't hold '__name__'
#         cls.__module__ = None
#
#     sentinel._cache[name] = Sentinel  # cache result
#     return Sentinel
#
#
# sentinel._cache = {}
#
# # 字典键值对转换
# def _invert(d):
#     return dict(zip(d.values(), d.keys()))
#
# handler = StreamHandler(sys.stdout, format_string=" | {record.message}")
# logger = Logger(__name__)
# logger.handlers.append(handler)
#
# if not csvdir:
#     csvdir = environ.get('CSVDIR')
#     if not csvdir:
#         raise ValueError("CSVDIR environment variable is not set")
#
# if not os.path.isdir(csvdir):
#     raise ValueError("%s is not a directory" % csvdir)


# def maybe_create_close_position_transaction(self, asset):
#     """强制平仓机制 --- 持仓特定标的的仓位"""
#     raise NotImplementedError('automatic operation')

# def manual_withdraw_operation(self, assets):
#     """
#         self.position_tracker.maybe_create_close_position_transaction
#         self.process_transaction(txn)
#     """
#     warnings.warn('avoid interupt automatic process')
#     self.position_tracker.maybe_create_close_position_transaction(assets)

# def copy_process_env(self):
#     """为子进程拷贝主进程中的设置执行，在add_process_env_sig装饰器中调用，外部不应主动使用"""
#     for module in self.register_module():
#         # 迭代注册了的需要拷贝内存设置的模块, 筛选模块中以g_或者_g_开头的, 且不能callable，即不是方法
#         sig_env = list(filter(
#             lambda sig: not callable(sig) and (sig.startswith('g_') or sig.startswith('_g_')), dir(module)))
#         module_name = module.__name__
#         for _sig in sig_env:
#             # 格式化类变量中对应模块属性的key
#             name = '{}_{}'.format(module_name, _sig)
#             # 根据应模块属性的key（name）getattr获取属性值
#             val = getattr(self, name)
#             # 为子模块内存变量进行值拷贝
#             module.__dict__[_sig] = val

# def add_process_env_sig(func):
#     """
#     初始化装饰器时给被装饰函数添加env关键字参数，在wrapper中将env对象进行子进程copy
#     由于要改方法签名，多个装饰器的情况要放在最下面
#     :param func:
#     :return:
#     """
#     # 获取原始函数参数签名，给并行方法添加env参数
#     sig = signature(func)
#
#     if 'env' not in list(sig.parameters.keys()):
#         parameters = list(sig.parameters.values())
#         # 通过强制关键字参数，给方法加上env
#         parameters.append(Parameter('env', Parameter.KEYWORD_ONLY, default=None))
#         # wrapper的__signature__进行替换
#         wrapper.__signature__ = sig.replace(parameters=parameters)
#
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # env = kwargs.pop('env', None)
#         if 'env' in kwargs:
#             """
#                 实际上linux, mac os上并不需要进行进程间模块内存拷贝，
#                 子进程fork后携带了父进程的内存信息，win上是需要的，
#                 暂时不做区分，都进行进程间的内存拷贝，如特别在乎效率的
#                 情况下基于linux系统，mac os可以不需要拷贝，如下：
#                 if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
#                     # 只有windows进行内存设置拷贝
#                     env.copy_process_env()
#             """
#             # if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
#             env = kwargs.pop('env', None)
#             if env is not None:
#                 # 将主进程中的env拷贝到子进程中
#                 env.copy_process_env()
#         return func(*args, **kwargs)
#
#     return wrapper


# def function_application(func):
#     """
#     Factory function for producing function application methods for Factor
#     subclasses.
#     """
#     if func not in NUMEXPR_MATH_FUNCS:
#         raise ValueError("Unsupported mathematical function '%s'" % func)
#
#     docstring = dedent(
#         """\
#         Construct a Factor that computes ``{}()`` on each output of ``self``.
#
#         Returns
#         -------
#         factor : zipline.pipe.Factor
#         """.format(func)
#     )
#
#     @with_doc(docstring)
#     @with_name(func)
#     def mathfunc(self):
#         if isinstance(self, NumericalExpression):
#             return NumExprFactor(
#                 "{func}({expr})".format(func=func, expr=self._expr),
#                 self.inputs,
#                 dtype=float64_dtype,
#             )
#         else:
#             return NumExprFactor(
#                 "{func}(x_0)".format(func=func),
#                 (self,),
#                 dtype=float64_dtype,
#             )
#     return mathfunc


# @singleton
# class Ignore(object):
#     def __str__(self):
#         return 'Argument.ignore'
#     __repr__ = __str__
#
#


# if g_is_ipython and not g_is_py3:
#     """ipython在python2的一些版本需要reload logging模块，否则不显示log信息"""
#     # noinspection PyUnresolvedReferences, PyCompatibility
#     reload(logging)
#     # pass
# from distutils.version import StrictVersion
#
#
# pandas_version = StrictVersion(pd.__version__)
# new_pandas = pandas_version >= StrictVersion('0.19')
# if pandas_version >= StrictVersion('0.20'):
#     def normalize_date(dt):
#         """
#         Normalize datetime.datetime value to midnight. Returns datetime.date as
#         a datetime.datetime at midnight
#
#         Returns
#         -------
#         normalized : datetime.datetime or Timestamp
#         """
#         return dt.normalize()
# else:
#     from pandas.tseries.tools import normalize_date
#


# class Event(object):
#
#     def __init__(self, initial_values=None):
#         if initial_values:
#             self.__dict__.update(initial_values)
#
#     def keys(self):
#         return self.__dict__.keys()
#
#     def __eq__(self, other):
#         return hasattr(other, '__dict__') and self.__dict__ == other.__dict__
#
#     def __contains__(self, name):
#         return name in self.__dict__
#
#     def __repr__(self):
#         return "Event({0})".format(self.__dict__)
#
#     def to_series(self, index=None):
#         return pd.Series(self.__dict__, index=index)


# def __hash__(self):
#     return id(self)
#
#
# def __contains__(self, column):
#     return column in self._table_expressions
#
#
# def __getitem__(self, column):
#     return self._table_expressions[column]
#
#
# def __iter__(self):
#     return iter(self._table_expressions)
#
#
# def __len__(self):
#     return len(self._table_expressions)
#
#
# def __call__(self, column):
#     if column in self:
#         return self
#     raise KeyError(column)

# exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# code = compile(self.algoscript, algo_filename, 'exec')
# exec_(code, self.namespace)
#
# def noop(*args, **kwargs):
#     pass
#
#

# from graphviz import Digraph,Graph
#
# h = Graph('hello', format='svg')
#
# h.edge('Hello', 'World')
#
# print(h.pipe().decode('utf-8'))



# import os, glob
#
# p = os.path.abspath('__file__')
# print('p', p)
# dir = os.path.dirname(p)
# print('dir', dir)
# base = os.path.basename(p)
# print('base', base)
# p_dir = os.getcwd()
# print('now directory', p_dir)
# c_test = os.path.split(os.getcwd())
# print('c_test', c_test)
# target = os.path.join(os.path.split(os.getcwd())[0], 'strat')
# print('target', target)
# files = os.path.join(target, '*')
# p = glob.glob(target + os.sep + 'cross.py')
# print('p', p)
# files = glob.glob('strat/*')
# print('files', files)

#
# def protect(cls):
#     def handler(signum, frame):
#         print(signum, frame)
#         raise SystemExit
#
#     while True:
#         # signal.signal(signal.SIGINT, handler)
#         signal.signal(signal.SIGINT, signal.SIG_IGN)
#         # signal.signal(signal.SIGINT, signal.SIG_DFL)
#
# def protect(cls):
#     def handler(signum, frame):
#         print(signum, frame)
#         print('now time', time.time())
#         time.sleep(10)
#         # raise SystemExit
#
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(1)
#     while True:
#         print('c_test')
#
# def protect(cls):
#     # Define signal handler function
#     def myHandler(signum, frame):
#         print('I received: ', signum)
#
#     # register signal.SIGTSTP's handler
#     signal.signal(signal.SIGINT, myHandler)
#     signal.pause()
#     print('End of Signal Demo')
