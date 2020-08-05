# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

import threading

context = threading.local()

def get_algo_instance():
    return getattr(context, 'algorithm', None)

def set_algo_instance(algo):
    context.algorithm = algo


class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    """
    def __init__(self, algo_instance):
        self.algo_instance = algo_instance

    def __enter__(self):
        """
        Set the given algo instance, storing any previously-existing instance.
        """
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        """
        Restore the algo instance stored in __enter__.
        """
        set_algo_instance(self.old_algo_instance)
