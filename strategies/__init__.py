# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import abstractmethod, ABC
# indicator --- terms --- pipe


class Strategy(ABC):

    @abstractmethod
    def compute(self, inputs, data):
        """
        :param inputs: assets list
        :param data: data needed to compute strategy
        :return: priority assets list
        """
        raise  NotImplementedError()
