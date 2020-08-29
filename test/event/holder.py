# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, re, pandas as pd
from gateway.spider import Crawler
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url
from gateway.spider.url import ASSET_FUNDAMENTAL_URL, HolderFields


class HolderWriter(Crawler):

    def writer(self):
        """股票增持、减持、变动情况"""
        deadline = self._retrieve_deadline_from_sqlite('holder')
        print(deadline)
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['holder'] % page
            raw = _parse_url(url, bs=False)
            match = re.search('\[(.*.)\]', raw)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            frame = pd.DataFrame(data, columns=HolderFields)
            frame.loc[:, 'sid'] = frame['代码']
            # '' -- 0.0
            frame.replace(to_replace='', value=0.0, inplace=True)
            print(frame)
            holdings = frame[frame['declared_date'] > deadline] if deadline else frame
            if len(holdings) == 0:
                break
            db.writer('holder', holdings)
            page = page + 1


if __name__ == '__main__':

    w = HolderWriter()
    w.writer()
