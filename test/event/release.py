# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from gateway.database.db_writer import db
from gateway.spider.url import ASSET_FUNDAMENTAL_URL
from gateway.spider import Crawler
from gateway.driver.tools import _parse_url


class ReleaseWriter(Crawler):

    def writer(self, s_date, e_date):
        """
            获取A股解禁数据
        """
        page = 1
        deadline = self._retrieve_deadline_from_sqlite('release')
        while True:
            url = ASSET_FUNDAMENTAL_URL['release'].format(page=page, start=s_date, end=e_date)
            text = _parse_url(url, encoding=None, bs=False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                data = [[item['gpdm'], item['ltsj'], item['xsglx'], item['zb']] for item in info]
                # release_date --- declared_date
                frame = pd.DataFrame(data, columns=['sid', 'release_date', 'release_type', 'zb'])
                frame.loc[:, 'declared_date'] = frame['release_date'].apply(lambda x: str(x)[:10])
                frame.dropna(axis=0, how='any', inplace=True)
                release = frame[frame['declared_date'] > deadline] if deadline else frame
                db.writer('unlock', release)
                page = page + 1
            else:
                break


if __name__ == '__main__':

    w = ReleaseWriter()
    w.writer('2020-08-25', '2020-08-28')
