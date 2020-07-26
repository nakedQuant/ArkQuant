# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC , abstractmethod
import numpy as np


class CancelPolicy(ABC):

    @abstractmethod
    def should_cancel(self,order):
        raise NotImplementedError


class ComposedCancel(CancelPolicy):
    """
     compose two rule with some composing function
    """
    def __init__(self,first,second):
        if not np.all(isinstance(first,CancelPolicy) and isinstance(second,CancelPolicy)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second

    def should_trigger(self,order):

        return self.first.should_cancel(order) & self.second.should_cancel(order)

