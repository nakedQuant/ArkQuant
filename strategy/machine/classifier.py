# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np
from functools import partial

int64_dtype = np.dtype('int64')
METHOD_OF_CATEGORY = frozenset(['bins', 'quantiles'])


def validate_type_method(raw, kwargs):
    if len(np.unique(raw.dtypes)) - 1:
        raise TypeError('the dtype of raw must be the same')
    if kwargs['method'] in METHOD_OF_CATEGORY and kwargs['bins']:
        assert True, ('expect method in %r,but received %s'
                      % (METHOD_OF_CATEGORY, kwargs['method']))


class Classifier(object):

    missing_value = [None, np.nan, np.NaN, np.NAN]

    def __setattr__(self, key, value):
        raise NotImplementedError

    # 基于integer encode
    def auto_encode(self, raw, **kwargs):
        validate_type_method(raw, kwargs)
        # Return a contiguous flattened array
        non_unique = set(np.ravel(raw)) - set(self.missing_value)
        bin = kwargs['bins']
        encoding = pd.cut(non_unique, bins=bin, labes=range(len(bin)))
        return zip(non_unique, encoding)

    @staticmethod
    # func --- integer
    def custom_encode(data, encoding_function, **kwargs):
        if kwargs:
            encoding_map = partial(encoding_function, **kwargs)
        else:
            encoding_map = encoding_function
        # otypes --- type
        encoding = np.vectorize(encoding_map, otypes=[int64_dtype])(data)
        return zip(data, encoding)
