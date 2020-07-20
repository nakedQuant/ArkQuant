"""
HDF5 Pricing File Format
------------------------
At the top level, the file is keyed by country (to support regional
files containing multiple countries).

Within each country, there are 3 subgroups:

``/data``
^^^^^^^^^
Each field (OHLCV) is stored in a dataset as a 2D array, with a row per
sid and a column per session. This differs from the more standard
orientation of dates x sids, because it allows each compressed block to
contain contiguous values for the same sid, which allows for better
compression.

.. code-block:: none

   /data
     /open
     /high
     /low
     /close
     /volume

``/index``
^^^^^^^^^^
Contains two datasets, the index of sids (aligned to the rows of the
OHLCV 2D arrays) and index of sessions (aligned to the columns of the
OHLCV 2D arrays) to use for lookups.

.. code-block:: none

   /index
     /sid
     /day

``/lifetimes``
^^^^^^^^^^^^^^
Contains two datasets, start_date and end_date, defining the lifetime
for each asset, aligned to the sids index.

.. code-block:: none

   /lifetimes
     /start_date
     /end_date

Example
^^^^^^^
Sample layout of the full file with multiple countries.

.. code-block:: none

   |- /equity
   |  |- /data
   |  |  |- /open
   |  |  |- /high
   |  |  |- /low
   |  |  |- /close
   |  |  |- /volume
   |  |
   |  |- /index
   |  |  |- /sid
   |  |  |- /day
   |  |
   |  |- /lifetimes
   |     |- /start_date
   |     |- /end_date
   |
   |- /fund
      |- /data
      |  |- /open
      |  |- /high
      |  |- /low
      |  |- /close
      |  |- /volume
      |
      |- /index
      |  |- /sid
      |  |- /day
      |
      |- /lifetimes
         |- /start_date
         |- /end_date
"""

from functools import partial
from itertools import chain
import numpy as np , pandas as pd ,h5py , tables


DEFAULT_SCALING_FACTORS = {
    # Retain 3 decimal places for prices.
    'open': 1000,
    'high': 1000,
    'low': 1000,
    'close': 1000,
    # Volume is expected to be a whole integer.
    'volume': 1,
}

def convert_price_with_scaling_factor(a, scaling_factor):
    conversion_factor = (1.0 / scaling_factor)

    zeroes = (a == 0)
    return np.where(zeroes, np.nan, a.astype('float64')) * conversion_factor

def coerce_to_uint32(a, scaling_factor):
    """
    Returns a copy of the array as uint32, applying a scaling factor to
    maintain precision if supplied.
    """
    return (a * scaling_factor).round().astype('uint32')

def check_indexes_all_same(indexes, message="Indexes are not equal."):
    """Check that a list of Index objects are all equal.

    Parameters
    ----------
    indexes : iterable[pd.Index]
        Iterable of indexes to check.

    Raises
    ------
    ValueError
        If the indexes are not all the same.
    """
    iterator = iter(indexes)
    first = next(iterator)
    for other in iterator:
        same = (first == other)
        if not same.all():
            #返回非0元素的位置
            bad_loc = np.flatnonzero(~same)[0]
            raise ValueError(
                "{}\nFirst difference is at index {}: "
                "{} != {}".format(
                    message, bad_loc, first[bad_loc], other[bad_loc]
                ),
            )

def days_and_sids_for_frames(frames):
    """
    Returns the date index and sid columns shared by a list of dataframes,
    ensuring they all match.

    Parameters
    ----------
    frames : dict {sid:pd.DataFrame}
        A list of dataframes indexed by day, with a column per sid.

    Returns
    -------
    days : np.array[datetime64[ns]]
        The days in these dataframes.
    sids : np.array[int64]
        The sids in these dataframes.

    Raises
    ------
    ValueError
        If the dataframes passed are not all indexed by the same days
        and sids.
    """
    if not frames:
        days = np.array([], dtype='datetime64[ns]')
        sids = np.array([], dtype='int64')
        return days, sids

    # Ensure the columns all match.
    check_indexes_all_same(
        [frame.columns for frame in frames.values()],
        message='Frames have mismatched sids.',
    )
    days = set(chain(*[frame.index for frame in frames.values()]))
    sids = set(frames)
    cols = frames.values()[0].columns
    return days,sids,cols


class HDF5DailyBarWriter(object):
    """
    Class capable of writing daily OHLCV data to disk in a format that
    can be read efficiently by HDF5DailyBarReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    date_chunk_size : int
        The number of days per chunk in the HDF5 file. If this is
        greater than the number of days in the data, the chunksize will
        match the actual number of days.

    See Also
    --------
    zipline.data.hdf5_daily_bars.HDF5DailyBarReader
    """
    def __init__(self, filename, date_chunk_size):
        self._filename = filename
        self._date_chunk_size = date_chunk_size

    def h5_file(self, mode):
        return h5py.File(self._filename, mode)

    def write(self,asset_type,frames,scaling_factors=None):
        """Write the OHLCV data for one country to the HDF5 file.

        Parameters
        ----------
        asset_type : str
            equity  or fund or convertible
        frames : dict[str, pd.DataFrame]
            A dict mapping each OHLCV field to a dataframe with a row
            for each date and a column for each sid. The dataframes need
            to have the same index and columns.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        # Add id to the index, so the frame is indexed by (date, id).
        # ohlcv_frame.set_index(sid_ix, append=True, inplace=True)

        if scaling_factors is None:
            scaling_factors = DEFAULT_SCALING_FACTORS

        with self.h5_file(mode='a') as h5_file:
            # ensure that the file version has been written
            h5_file.attrs['version'] = 'version'
            #多维度创建group
            category_group = h5_file.create_group(asset_type)
            category_group.attrs['scaling_factor'] = scaling_factors
            data_group = category_group.create_group('data')
            index_group = category_group.create_group('index')
            # Note that this functions validates that all of the frames
            # share the same days and sids.
            days, sids,fields = days_and_sids_for_frames(frames)
            # Write sid and date indices.
            index_group.create_dataset('sid', data=sids)
            # h5py does not support datetimes, so they need to be stored
            # as integers.
            index_group.create_dataset('day', data=days.astype(np.int64))
            index_group.create_dataset('field',data = fields)
            #
            for sid , frame in frames.items():
                frame.sort_index(inplace=True)
                data_group.create_dataset(sid,
                                          compression='lzf',
                                          shuffle=True,
                                          data=frame,
                                          )


class HDF5DailyBarReader(object):
    """
    Parameters
    ---------
    country_group : h5py.Group
        The group for a single country in an HDF5 daily pricing file.
    """
    def __init__(self, country_group):
        self._country_group = country_group

        self._postprocessors = {
            'open': partial(convert_price_with_scaling_factor,
                          scaling_factor=self._read_scaling_factor(OPEN)),
            'high': partial(convert_price_with_scaling_factor,
                          scaling_factor=self._read_scaling_factor(HIGH)),
            'low': partial(convert_price_with_scaling_factor,
                         scaling_factor=self._read_scaling_factor(LOW)),
            'close': partial(convert_price_with_scaling_factor,
                           scaling_factor=self._read_scaling_factor(CLOSE)),
            'volume': lambda a: a,
        }

    @classmethod
    def from_file(cls, h5_file):
        """
        Construct from an h5py.File and a country code.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        if h5_file.attrs['version'] != VERSION:
            raise ValueError(
                'mismatched version: file is of version %s, expected %s' % (
                    h5_file.attrs['version'],
                    VERSION,
                ),
            )

        return cls(h5_file)

    @classmethod
    def from_path(cls, path):
        """
        Construct from a file path and a country code.

        Parameters
        ----------
        path : str
            The path to an HDF5 daily pricing file.
        """
        return cls.from_file(h5py.File(path))

    def _read_scaling_factor(self, field):
        return self._category[DATA][field].attrs[SCALING_FACTOR]

    def load_raw_arrays(self,
                        sids,
                        start_date,
                        end_date,
                        assets):
        """
        Parameters
        ----------
        sids : list of str
                sids
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """

        start = start_date.asm8
        end = end_date.asm8
        date_slice = self._compute_date_range_slice(start, end)
        n_dates = date_slice.stop - date_slice.start

        # Create a buffer into which we'll read data from the h5 file.
        # Allocate an extra row of space that will always contain null values.
        # We'll use that space to provide "data" for entries in ``assets`` that
        # are unknown to us.
        full_buf = np.zeros((len(self.sids) + 1, n_dates), dtype=np.uint32)
        # We'll only read values into this portion of the read buf.
        mutable_buf = full_buf[:-1]

        # Indexer that converts an array aligned to self.sids (which is what we
        # pull from the h5 file) into an array aligned to ``assets``.
        #
        # Unknown assets will have an index of -1, which means they'll always
        # pull from the last row of the read buffer. We allocated an extra
        # empty row above so that these lookups will cause us to fill our
        # output buffer with "null" values.
        sid_selector = self._make_sid_selector(assets)

        out = []
        for column in columns:
            # Zero the buffer to prepare to receive new data.
            mutable_buf.fill(0)

            dataset = self._category[DATA][column]

            # Fill the mutable portion of our buffer with data from the file.
            dataset.read_direct(
                mutable_buf,
                np.s_[:, date_slice],
            )

            # Select data from the **full buffer**. Unknown assets will pull
            # from the last row, which is always empty.
            out.append(self._postprocessors[column](full_buf[sid_selector].T))

        return out

    def _make_sid_selector(self, assets):
        """
        Build an indexer mapping ``self.sids`` to ``assets``.

        Parameters
        ----------
        assets : list[int]
            List of assets requested by a caller of ``load_raw_arrays``.

        Returns
        -------
        index : np.array[int64]
            Index array containing the index in ``self.sids`` for each location
            in ``assets``. Entries in ``assets`` for which we don't have a sid
            will contain -1. It is caller's responsibility to handle these
            values correctly.
        """
        assets = np.array(assets)
        sid_selector = self.sids.searchsorted(assets)
        #查找相同的列，invert = True
        unknown = np.in1d(assets, self.sids, invert=True)
        sid_selector[unknown] = -1
        return sid_selector

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
        self._validate_assets([sid])
        self._validate_timestamp(dt)

        sid_ix = self.sids.searchsorted(sid)
        dt_ix = self.dates.searchsorted(dt.asm8)

        value = self._postprocessors[field](
            self._category[DATA][field][sid_ix, dt_ix]
        )
        if np.isnan(value):
            if dt.asm8 < self.asset_start_dates[sid_ix]:
                raise NoDataBeforeDate()

            if dt.asm8 > self.asset_end_dates[sid_ix]:
                raise NoDataAfterDate()

        return value


class H5MinuteWriter(object):
    """
    Writer for files containing minute bar updates for consumption by a writer
    for a ``MinuteBarReader`` format.

    Parameters
    ----------
    path : str
        The destination path.
    complevel : int, optional
        The HDF5 complevel, defaults to ``5``.
    complib : str, optional
        The HDF5 complib, defaults to ``zlib``.
    """

    _COMPLEVEL = 5
    _COMPLIB = 'zlib'

    def __init__(self, path, complevel=None, complib=None):
        self._complevel = complevel if complevel \
            is not None else self._COMPLEVEL
        self._complib = complib if complib \
            is not None else self._COMPLIB
        self._path = path

    def write(self, frames):
        """
        Write the frames to the target HDF5 file, using the format used by
        ``pd.Panel.to_hdf``

        Parameters
        ----------
        frames : iter[(int, DataFrame)] or dict[int -> DataFrame]
            An iterable or other mapping of sid to the corresponding OHLCV
            pricing data.
        """
        with pd.HDFStore(self._path, 'w',
                      complevel=self._complevel, complib=self._complib) \
                as store:
            panel = pd.Panel.from_dict(dict(frames))
            panel.to_hdf(store, 'updates')
        with tables.open_file(self._path, mode='r+') as h5file:
            h5file.set_node_attr('/', 'version', 0)


class H5MinuteReader(object):
    """
    Reader for minute bar updates stored in HDF5 files.

    Parameters
    ----------
    path : str
        The path of the HDF5 file from which to source data.
    """
    def __init__(self, path):
        self._panel = pd.read_hdf(path)

    def read(self, dts, sids):
        panel = self._panel[sids, dts, :]
        return panel.iteritems()