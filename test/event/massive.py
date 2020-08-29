# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, pandas as pd
from gateway.database.db_writer import db
from gateway.spider import Crawler
from gateway.driver.tools import _parse_url
from gateway.spider.url import ASSET_FUNDAMENTAL_URL, MassiveFields


# NullMassiveFields = ['sid', 'declared_date', 'discount','bid_price',
#                      'bid_volume', 'buyer', 'seller', 'cjeltszb']


class MassiveWriter(Crawler):

    def writer(self, s_date, e_date):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        page = 1
        deadline = self._retrieve_deadline_from_sqlite('massive')
        print('deadline', deadline)
        while True:
            url = ASSET_FUNDAMENTAL_URL['massive'].format(page=page, start=s_date, end=e_date)
            data = _parse_url(url, bs=False, encoding=None)
            data = json.loads(data)
            if data['data'] and len(data['data']):
                frame = pd.DataFrame(data['data'])
                frame.rename(columns=MassiveFields, inplace=True)
                frame.loc[:, 'declared_date'] = frame['trade_dt'].apply(lambda x: str(x)[:10])
                # frame.dropna(axis=0, how='all', subset=NullMassiveFields, inplace=True)
                frame.dropna(axis=0, how='any', inplace=True)
                massive = frame[frame['declared_date'] > deadline] if deadline else frame
                db.writer('massive', massive)
                page = page + 1
            else:
                break


if __name__ == '__main__':

    w = MassiveWriter()
    w.writer('2020-08-25', '2020-08-28')


