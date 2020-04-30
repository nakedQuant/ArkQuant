#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""

from abc import ABC,abstractmethod
import datetime

class Ancestor(ABC):

    frequency = None
    nowdays = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

    @classmethod
    def _init_db(cls,init = False):
        if init:
            DataLayer.initialize()
        cls.db = DataLayer()

    def switch_mode(self):
        """决定daily or init"""
        if self.frequency:
            lmt = 1
        else:
            lmt = 10000
        return lmt

    @abstractmethod
    def _get_prefix(self):

        raise NotImplementedError

    @abstractmethod
    def _download_assets(self):

        raise NotImplementedError

    @abstractmethod
    def _download_kline(self):

        raise NotImplementedError

    @abstractmethod
    def _run_session(self):

        raise NotImplementedError

    @abstractmethod
    def run_bulks(self):

        raise NotImplementedError