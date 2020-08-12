# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from sqlalchemy import MetaData
from gateWay.driver import engine


class Crawler(ABC):

    @property
    def engine(self):
        return engine

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


__all__ = [Crawler]