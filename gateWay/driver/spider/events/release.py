# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from gateWay.driver.tools import _parse_url
from gateWay.driver.db_writer import db
from gateWay.driver.spider.xml import ASSET_FUNDAMENTAL_URL
from gateWay.driver.spider import Crawler


class ReleaseWriter(Crawler):

    @classmethod
    def writer(cls, s_date, e_date):
        """
            获取A股解禁数据
        """
        count = 1
        prefix = '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(s_date, e_date) + \
                 '&js={"data":(x)}'
        while True:
            url = ASSET_FUNDAMENTAL_URL['release'] % count + prefix
            text = _parse_url(url, encoding=None,bs=False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'], item['ltsj'],item['xsglx'], item['zb']] for item in info]
                release = pd.DataFrame(raw, columns=['sid', 'release_date', 'release_type', 'cjeltszb'])
                db.writer('release', release)
                count = count + 1
            else:
                break
