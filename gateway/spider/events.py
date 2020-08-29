# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, re, pandas as pd
from threading import Thread
from gateway.spider import Crawler
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url
from gateway.spider.url import ASSET_FUNDAMENTAL_URL

__all__ = ['EventWriter']

EVENTS = frozenset(['holder', 'massive', 'release'])

# holder
HolderFields = ['代码', '中文', '现价','涨幅', '股东', '方式', '变动股本', '占总流通比', '途径', '总持仓',
                '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', 'declared_date']

# massive
MassiveFields = {'TDATE': 'trade_dt',
                 'SECUCODE': 'sid',
                 'SNAME': 'name',
                 'PRICE': 'bid_price',
                 'TVOL': 'bid_volume',
                 'TVAL': 'amount',
                 'BUYERCODE': 'buyer_code',
                 'BUYERNAME': 'buyer',
                 'SALESCODE': 'seller_code',
                 'SALESNAME': 'seller',
                 'Stype': 'stype',
                 'Unit': 'unit',
                 'RCHANGE': 'pct',
                 'CPRICE': 'close',
                 'YSSLTAG': 'YSSLTAG',
                 'Zyl': 'discount',
                 'Cjeltszb': 'cjeltszb',
                 'RCHANGE1DC': '1_pct',
                 'RCHANGE5DC': '5_pct',
                 'RCHANGE10DC': '10_pct',
                 'RCHANGE20DC': '20_pct'}


class EventWriter(Crawler):

    def _writer_holder(self, *args):
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

    def _writer_massive(self, s_date, e_date):
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
                frame.dropna(axis=0, how='any', inplace=True)
                massive = frame[frame['declared_date'] > deadline] if deadline else frame
                print('massive', massive)
                db.writer('massive', massive)
                page = page + 1
            else:
                break

    def _writer_release(self, s_date, e_date):
        """
            获取A股解禁数据
        """
        page = 1
        deadline = self._retrieve_deadline_from_sqlite('unfreeze')
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
                print('release', release)
                db.writer('unfreeze', release)
                page = page + 1
            else:
                break

    def writer(self, sdate, edate):
        threads = []
        for method_name in EVENTS:
            method = getattr(self, '_writer_%s' % method_name)
            thread = Thread(target=method, args=(sdate, edate), name=method)
            thread.start()
            threads.append(thread)

        for t in threads:
            print(t.name, t.is_alive())
            t.join()


if __name__ == '__main__':

    w = EventWriter()
    # date -- e.g '2020-02-08'
    w.writer('2020-08-25', '2020-08-27')
