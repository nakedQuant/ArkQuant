"""
classifier.py
"""
import pandas as pd,numpy as np
from functools import partial

int64_dtype = np.dtype('int64')
bool_dtype = np.dtype('bool')
object_dtype = np.dtype('O')
categorical_dtype = object_dtype

METHOD_OF_CATEGORY = frozenset(['bins','quantiles'])

def validate_type_method(raw,kwargs):
    if len(np.unique(raw.dtypes)) -1 :
        raise TypeError('the dtype of raw must be the same')
    if kwargs['method'] in METHOD_OF_CATEGORY and kwargs['bins']:
        assert True,('expect method in %r,but received %s'
                     %(METHOD_OF_CATEGORY, kwargs['method']))


class Classifier(object):

    missing_value = [None,np.nan,np.NaN,np.NAN]

    def __setattr__(self, key, value):
        raise NotImplementedError

    def auto_encode(self,
                        raw,
                        **kwargs):
        validate_type_method(raw,kwargs)
        non_unique = set(ravel(raw)) - set(self.missing_value)
        bin = kwargs['bins']
        encoding = pd.cut(non_unique,bins = bin,labes = range(len(bin)))
        return zip(non_unique,encoding)

    def custom_encode(self,
                        data,
                        encoding_function,**kwargs):
        if kwargs:
            encoding_map = partial(encoding_function,**kwargs)
        else:
            encoding_map = encoding_function
        encoding = np.vectorize(encoding_map, otypes=[int64_dtype])(data)
        return zip(data,encoding)