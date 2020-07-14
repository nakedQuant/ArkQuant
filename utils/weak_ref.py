"""
Tools for memoization of function results.
"""
from collections import OrderedDict, Sequence
from itertools import compress
#弱引用 以及沙盒函数需要研究一下 、gcc回收机制
from weakref import WeakKeyDictionary, ref

from toolz.sandbox import unzip


class _WeakArgs(Sequence):
    """
    Works with _WeakArgsDict to provide a weak cache for function args.
    When any of those args are gc'd, the pair is removed from the cache.
    """
    def __init__(self, items, dict_remove=None):
        def remove(selfref=ref(self), dict_remove=dict_remove):
            self = selfref()
            if self is not None and dict_remove is not None:
                dict_remove(self)

        self._items, self._selectors = unzip(self._try_ref(item, remove)
                                             for item in items)
        self._items = tuple(self._items)
        self._selectors = tuple(self._selectors)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    @staticmethod
    def _try_ref(item, callback):
        try:
            """ 
                Return a weak reference to object.
                If callback is provided and not None, and the returned weakref object is still alive, 
                the callback will be called when the object is about to be finalized; 
                the weak reference object will be passed as the only parameter to the callback; 
                the referent will no longer be available.
            """
            return ref(item, callback), True
        except TypeError:
            return item, False

    @property
    def alive(self):
        """
        itertools.compress(data, selectors)¶
        Make an iterator that filters elements from data returning only those
        that have a corresponding element in selectors that evaluates to True.
        """
        return all(item() is not None
                   for item in compress(self._items, self._selectors))

    def __eq__(self, other):
        return self._items == other._items

    def __hash__(self):
        try:
            return self.__hash
        except AttributeError:
            h = self.__hash = hash(self._items)
            return h


class _WeakArgsDict(WeakKeyDictionary, object):
    def __delitem__(self, key):
        del self.data[_WeakArgs(key)]

    def __getitem__(self, key):
        return self.data[_WeakArgs(key)]

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.data)

    def __setitem__(self, key, value):
        self.data[_WeakArgs(key, self._remove)] = value

    def __contains__(self, key):
        try:
            wr = _WeakArgs(key)
        except TypeError:
            return False
        return wr in self.data

    def pop(self, key, *args):
        return self.data.pop(_WeakArgs(key), *args)

