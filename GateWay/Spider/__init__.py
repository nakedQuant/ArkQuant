#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""

from abc import ABC,abstractmethod
import requests,datetime
from bs4 import BeautifulSoup
from GateWay.Driver import DataLayer

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

    @staticmethod
    def _parse_url(url,encoding = 'gbk',bs = True):
        Header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
        req = requests.get(url,headers = Header,timeout = 1)
        if encoding:
            req.encoding = encoding
        if bs:
            raw = BeautifulSoup(req.text, features='lxml')
        else:
            raw = req.text
        return raw

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

