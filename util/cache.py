# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import MutableMapping
from functools import partial
from distutils import dir_util
from shutil import rmtree, move
from tempfile import mkdtemp, NamedTemporaryFile
import os, pickle, errno, pandas as pd
from util.paths import ensure_directory


class DummyMapping(object):
    """
    Dummy object used to provide a mapping interface for singular values.
    """
    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        return self._value


class dataframe_cache(MutableMapping):
    """A disk-backed cache for dataframes.

    ``dataframe_cache`` is a mutable mapping from string names to pandas
    DataFrame objects.
    This object may be used as a context manager to delete the cache directory
    on exit.

    Parameters
    ----------
    path : str, optional
        The directory path to the cache. Files will be written as
        ``path/<keyname>``.
    lock : Lock, optional
        Thread lock for multithreaded/multiprocessed access to the cache.
        If not provided no locking will be used.
    clean_on_failure : bool, optional
        Should the directory be cleaned up if an exception is raised in the
        context manager.
    serialize : {'msgpack', 'pickle:<n>'}, optional
        How should the data be serialized. If ``'pickle'`` is passed, an
        optional pickle protocol can be passed like: ``'pickle:3'`` which says
        to use pickle protocol 3.

    Notes
    -----
    The syntax ``cache[:]`` will load all key:value pairs into memory as a
    dictionary.
    The cache uses a temporary file format that is subject to change between
    versions of zipline.
    """
    def __init__(self,
                 path=None,
                 lock=None,
                 clean_on_failure=True,
                 serialization='msgpack'):
        # create directory
        self.path = path if path is not None else mkdtemp()
        self.lock = lock if lock is not None else nop_context
        self.clean_on_failure = clean_on_failure

        if serialization == 'msgpack':
            self.serialize = pd.DataFrame.to_msgpack
            self.deserialize = pd.read_msgpack
            self._protocol = None
        else:
            s = serialization.split(':', 1)
            if s[0] != 'pickle':
                raise ValueError(
                    "'serialization' must be either 'msgpack' or 'pickle[:n]'",
                )
            self._protocol = int(s[1]) if len(s) == 2 else None

            self.serialize = self._serialize_pickle
            self.deserialize = partial(pickle.load, encoding='latin-1')

        ensure_directory(self.path)

    def _serialize_pickle(self, df, path):
        with open(path, 'wb') as f:
            pickle.dump(df, f, protocol=self._protocol)

    def _keypath(self, key):
        return os.path.join(self.path, key)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, tb):
        if not (self.clean_on_failure or value is None):
            # we are not cleaning up after a failure and there was an exception
            return

        with self.lock:
            # shutil rmtree --- delete an entile directory tree
            rmtree(self.path)

    def __getitem__(self, key):
        # self.items() return key value
        if key == slice(None):
            return dict(self.items())

        with self.lock:
            try:
                with open(self._keypath(key), 'rb') as f:
                    return self.deserialize(f)
            except IOError as e:
                if e.errno != errno.ENOENT:
                    raise
                raise KeyError(key)

    def __setitem__(self, key, value):
        with self.lock:
            self.serialize(value, self._keypath(key))

    def __delitem__(self, key):
        with self.lock:
            try:
                os.remove(self._keypath(key))
            except OSError as e:
                if e.errno == errno.ENOENT:
                    # raise a keyerror if this directory did not exist
                    raise KeyError(key)
                # reraise the actual oserror otherwise
                raise

    def __iter__(self):
        return iter(os.listdir(self.path))

    def __len__(self):
        return len(os.listdir(self.path))

    def __repr__(self):
        # repr 为函数
        return '<%s: keys={%s}>' % (
            type(self).__name__,
            ', '.join(map(repr, sorted(self))),
        )
