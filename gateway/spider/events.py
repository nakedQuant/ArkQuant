# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json, re, pandas as pd, time, numpy as np
from threading import Thread
from gateway.spider import Crawler
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url
from gateway.spider.url import ASSET_FUNDAMENTAL_URL

__all__ = ['EventWriter']

EVENTS = frozenset(['holder', 'massive', 'release', 'margin'])

# holder
HolderFields = ['代码', '中文', '现价','涨幅', '股东', '方式', '变动股本', '占总流通比', '途径', '总持仓',
                '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', 'declared_date']

# massive
MassiveFields = {'TDATE': 'declared_date',
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
        deadline = self._retrieve_deadlines_from_sqlite('holder')
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['holder'] % page
            try:
                text = _parse_url(url, bs=False)
            except Exception as e:
                print('error %r' % e)
                time.sleep(np.random.randint(5, 10))
                text = _parse_url(url, bs=False)
            match = re.search('\[(.*.)\]', text)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            frame = pd.DataFrame(data, columns=HolderFields)
            frame.loc[:, 'sid'] = frame['代码']
            # '' -- 0.0
            frame.replace(to_replace='', value=0.0, inplace=True)
            holdings = frame[frame['declared_date'] > deadline.max()] if not deadline.empty else frame
            if len(holdings) == 0:
                break
            db.writer('holder', holdings)
            page = page + 1

    def _writer_margin(self, *args):
        """获取市场全量融资融券"""
        deadline = self._retrieve_deadlines_from_sqlite('margin')
        print('deadline', deadline)
        page = 1
        while True:
            req_url = ASSET_FUNDAMENTAL_URL['margin'] % page
            try:
                raw = _parse_url(req_url, bs=False)
                raw = json.loads(raw)
            except Exception as e:
                print(e)
                time.sleep(np.random.randint(5, 10))
                raw = _parse_url(req_url, bs=False)
            if raw['result']['data']:
                raw = [[item['DIM_DATE'], item['RZYE'], item['RZYEZB'], item['RQYE']]
                       for item in raw['result']['data']]
                frame = pd.DataFrame(raw, columns=['declared_date', 'rzye', 'rzyezb', 'rqye'])
                frame['declared_date'] = frame['declared_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
                frame.loc[:, ['rzye', 'rqye']] = frame.loc[:, ['rzye', 'rqye']].div(1e8)
                frame.fillna(0.0, inplace=True)
                margin = frame[frame['declared_date'] > deadline] if deadline else frame
                print('marign', margin)
                db.writer('margin', margin)
                page = page + 1
            else:
                break

    def _writer_massive(self, s_date, e_date):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        deadline = self._retrieve_deadlines_from_sqlite('massive')
        print('deadline', deadline.max())
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['massive'].format(page=page, start=s_date, end=e_date)
            try:
                text = _parse_url(url, bs=False, encoding=None)
            except Exception as e:
                print('error %r' % e)
                time.sleep(np.random.randint(5, 10))
                text = _parse_url(url, bs=False)
            data = json.loads(text)
            if data['data'] and len(data['data']):
                frame = pd.DataFrame(data['data'])
                frame.rename(columns=MassiveFields, inplace=True)
                # frame.loc[:, 'declared_date'] = frame['declared_date'].apply(lambda x: str(x)[:10])
                frame['declared_date'] = frame['declared_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
                frame.dropna(axis=0, how='all', inplace=True)
                print('frame', frame)
                massive = frame[frame['declared_date'] > deadline.max()] if not deadline.empty else frame
                print('massive', massive)
                db.writer('massive', massive)
                page = page + 1
            else:
                break

    def _writer_release(self, s_date, e_date):
        """
            获取A股解禁数据
        """
        deadline = self._retrieve_deadlines_from_sqlite('unfreeze')
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['release'].format(page=page, start=s_date, end=e_date)
            try:
                text = _parse_url(url, encoding=None, bs=False)
            except Exception as e:
                print(e)
                time.sleep(np.random.randint(5, 10))
                text = _parse_url(url, bs=False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                data = [[item['gpdm'], item['ltsj'], item['xsglx'], item['zb']] for item in info]
                # release_date --- declared_date
                frame = pd.DataFrame(data, columns=['sid', 'declared_date', 'release_type', 'zb'])
                # frame.loc[:, 'declared_date'] = frame['release_date'].apply(lambda x: str(x)[:10])
                frame['declared_date'] = frame['declared_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
                frame.dropna(axis=0, how='all', inplace=True)
                print('frame', frame)
                release = frame[frame['declared_date'] > deadline.max()] if not deadline.empty else frame
                print('release', release)
                db.writer('unfreeze', release)
                page = page + 1
            else:
                break

    def _writer_internal(self, sdate, edate):
        threads = []
        for method_name in EVENTS:
            method = getattr(self, '_writer_%s' % method_name)
            thread = Thread(target=method, args=(sdate, edate), name=method)
            thread.start()
            threads.append(thread)

        for t in threads:
            print(t.name, t.is_alive())
            t.join()

    def writer(self, sdate, edate):
        self._writer_internal(sdate, edate)


if __name__ == '__main__':

    w = EventWriter()
    # w.writer('2010-01-01', '2020-09-03')
    w._writer_margin()
