# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from gateway.database.db_writer import db
from gateway.spider import Crawler
from gateway.spider.xml import ASSET_FUNDAMENTAL_URL, MassiveFields
from gateway.driver.tools import _parse_url


class MassiveWriter(Crawler):

    def writer(self, s_date, e_date):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        count = 1
        prefix = 'js={"data":(x)}&filter=(Stype=%27EQA%27)' + \
                 '(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(s_date, e_date)
        while True:
            url = ASSET_FUNDAMENTAL_URL['massive'] % count + prefix
            raw = _parse_url(url, bs=False, encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                massive = pd.DataFrame(raw['data'], columns=MassiveFields)
                db.writer('massive', massive)
                count = count + 1
            else:
                break
