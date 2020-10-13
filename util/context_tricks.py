# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from warnings import (
    catch_warnings,
    filterwarnings,
)
from contextlib import contextmanager


@object.__new__
class nop_context(object):
    """A nop context manager.
    """
    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


def _nop(*args, **kwargs):
    pass


class CallbackManager(object):
    """Create a context manager from a pre-execution callback and a
    post-execution callback.

    Parameters
    ----------
    pre : (...) -> any, optional
        A pre-execution callback. This will be passed ``*args`` and
        ``**kwargs``.
    post : (...) -> any, optional
        A post-execution callback. This will be passed ``*args`` and
        ``**kwargs``.

    Notes
    -----
    The enter value of this context manager will be the result of calling
    ``pre(*args, **kwargs)``
    """
    def __init__(self, pre=None, post=None):
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self, *args, **kwargs):
        return _ManagedCallbackContext(self.pre, self.post, args, kwargs)

    # special case, if no extra args are passed make this a context manager
    # which forwards no args to pre and post
    def __enter__(self):
        return self.pre()

    def __exit__(self, *excinfo):
        self.post()


class _ManagedCallbackContext(object):
    def __init__(self, pre, post, args, kwargs):
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self._pre(*self._args, **self._kwargs)

    def __exit__(self, *excinfo):
        self._post(*self._args, **self._kwargs)


class WarningContext(object):
    """
    Re-usable contextmanager for contextually managing warnings.
    """
    def __init__(self, *warning_specs):
        self._warning_specs = warning_specs
        self._catchers = []

    def __enter__(self):
        catcher = catch_warnings()
        catcher.__enter__()
        self._catchers.append(catcher)
        for args, kwargs in self._warning_specs:
            filterwarnings(*args, **kwargs)
        return self

    def __exit__(self, *exc_info):
        catcher = self._catchers.pop()
        return catcher.__exit__(*exc_info)


def ignore_nanwarnings():
    """
    Helper for building a WarningContext that ignores warnings from numpy's
    nanfunctions.
    """
    return WarningContext(
        (
            ('ignore',),
            {'category': RuntimeWarning, 'module': 'numpy.lib.nanfunctions'},
        )
    )


@contextmanager
def ignore_pandas_nan_categorical_warning():
    with catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in categories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipe, so for now just silence the warning.
        filterwarnings(
            'ignore',
            category=FutureWarning,
        )
        yield
