# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC , abstractmethod
import numpy as np


class CancelPolicy(ABC):
    """
        Abstract cancellation policy interface.
        cancel policy --- 过滤pipeline 里面的结果
    """
    @abstractmethod
    def should_cancel(self, event):
        """Should all open orders be cancelled?

        Parameters
        ----------
        event : enum-value
            An event type, one of:
              - :data:`zipline.gens.sim_engine.BAR`
              - :data:`zipline.gens.sim_engine.DAY_START`
              - :data:`zipline.gens.sim_engine.DAY_END`
              - :data:`zipline.gens.sim_engine.MINUTE_END`

        Returns
        -------
        should_cancel : bool
            Should all open orders be cancelled?
        """
        pass


class ComposedCancel(CancelPolicy):
    """
     compose two rule with some composing function
    """
    def __init__(self, first, second):
        if not np.all(isinstance(first, CancelPolicy) and isinstance(second, CancelPolicy)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second

    def should_cancel(self, order):

        return self.first.should_cancel(order) & self.second.should_cancel(order)


class EODCancel(CancelPolicy):
    """This policy cancels open orders at the end of the day.  For now,
    Zipline will only apply this policy to minutely simulations.

    Parameters
    ----------
    warn_on_cancel : bool, optional
        Should a warning be raised if this causes an order to be cancelled?
    """
    def __init__(self, warn_on_cancel=True):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, event):
        return event == SESSION_END


class NeverCancel(CancelPolicy):
    """Orders are never automatically canceled.
    """
    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, event):
        return False


class LimitCancel(CancelPolicy):

    def __init__(self, limit = 0.0990):
        self.default_limit = limit

    def should_cancel(self, event):
        raise NotImplementedError()