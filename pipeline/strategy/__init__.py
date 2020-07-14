# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from abc import ABC, abstractmethod

class AlgoBase(ABC):

    @abstractmethod
    def compute(self,inputs):

        raise NotImplementedError('computing core')
