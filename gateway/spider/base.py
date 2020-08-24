# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from sqlalchemy import MetaData
from gateway.database import engine
from gateway.driver.tools import _parse_url

__all__ = ['Crawler']


class Crawler(ABC):

    engine = engine

    @property
    def tool(self):
        return _parse_url

    @property
    def metadata(self):
        return MetaData(bind=self.engine)

    @abstractmethod
    def writer(self, *args):
        """
            intend to spider data from online
        :return:
        """
        raise NotImplementedError()


